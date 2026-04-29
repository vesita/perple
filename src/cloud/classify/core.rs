use crate::{
    cloud::{
        CldBud,
        classify::{claster::Claster, environment::*},
    },
    swapl::global_swapl,
    utils::stream::{Cream, Eap, Stream},
};

#[derive(Debug)]
pub enum ClassifyError {
    Error,
}

pub struct Classify {
    cream: Cream<Vec<[f32; 3]>, Vec<CldBud>>,
    claster: Claster,
    ground_plane_out: Eap<Stream<[f32; 4]>>,
}

impl Classify {
    pub fn new() -> Self {
        let swapl = global_swapl();
        Self {
            cream: Cream {
                in_stream: swapl.clouds_out.clone(),
                out_stream: swapl.cld_objs.clone(),
            },
            claster: Claster::new(),
            ground_plane_out: swapl.ground_plane.clone(),
        }
    }

    pub fn act(&mut self) -> Result<(), ClassifyError> {
   let mut target = {
       let mut stream = self.cream.in_stream.blocking_lock();
            match stream.read() {
               Some(target) => target,
                None => return Ok(()), // 没有数据可处理
            }
        };

   let(slice_index, grounds, plane_eq) = single_pick_ground(&mut target);
       println!("完成地面提取，已过滤{}个点", slice_index);
       if let Some(eq) = plane_eq {
           let mut gp = self.ground_plane_out.blocking_lock();
           let _ = gp.write(eq);
       }
        // let (slice_index, walls) = pick_wall(&mut target[slice_index..]);
        // println!("完成墙壁提取，已过滤{}个点", slice_index);
    // 优化：直接传递切片引用，避免不必要的 to_vec() 克隆
   let _ = self.claster.claster(&target[slice_index..]);
   let targets = self.claster.to_cldbuds();

        // 合并所有的 CldBud 到一个 Vec 中
   let mut all_buds = grounds;
        // all_buds.extend(walls);
        all_buds.extend(targets);

        {
       let mut stream = self.cream.out_stream.blocking_lock();
        let _ = stream.write(all_buds);
        }
       Ok(())
    }
}
