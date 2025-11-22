use std::sync::{Arc, Mutex};
use pcd_rs::{DynReader, DynRecord};

use crate::utils::stream::Stream;


pub struct Lidar {
    input_stream: Arc<Mutex<Stream<Option<DynRecord>>>>,
}


impl Lidar {

}