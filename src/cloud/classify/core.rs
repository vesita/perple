use crate::{cloud::{CldBud, classify::{claster::Claster, environment::*}}, swapl::global_swapl, utils::stream::Cream};


pub enum ClassifyError {
    Error
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
        if let Some(mut target) = self.cream.read() {
            let (slice_index, grounds) = single_pick_ground(&mut target);
            println!("完成地面提取，已过滤{}个点", slice_index);
            // let (slice_index, walls) = pick_wall(&mut target[slice_index..]);
            // println!("完成墙壁提取，已过滤{}个点", slice_index);
            self.claster.claster(&target[slice_index..].to_vec());
            let targets = self.claster.to_cldbuds();

            // 合并所有的CldBud到一个Vec中
            let mut all_buds = grounds;
            // all_buds.extend(walls);
            all_buds.extend(targets);

            let _ = self.cream.write(all_buds);
        }
        Ok(())
    }
}