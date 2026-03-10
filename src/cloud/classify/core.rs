use crate::{
    cloud::{
        CldBud,
        classify::{claster::Claster, environment::*},
    },
    swapl::global_swapl,
    utils::stream::Cream,
};

pub enum ClassifyError {
    Error,
}

pub struct Classify {
    cream: Cream<Vec<[f32; 3]>, Vec<CldBud>>,
    claster: Claster,
}

impl Classify {
    pub fn new() -> Self {
        let swapl = global_swapl();
        Self {
            cream: Cream {
                in_stream: swapl.cloud_in_world.clone(),
                out_stream: swapl.cld_objs.clone(),
            },
            claster: Claster::new(),
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

    let (slice_index, grounds) = single_pick_ground(&mut target);
        println!("完成地面提取，已过滤{}个点", slice_index);
        // let (slice_index, walls) = pick_wall(&mut target[slice_index..]);
        // println!("完成墙壁提取，已过滤{}个点", slice_index);
    let _ = self.claster.claster(&target[slice_index..].to_vec());
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
