use std::sync::Arc;
use bevy::prelude::*;

use crate::{Swapl, cloud::core::Lidar};
use std::sync::Mutex;

#[derive(Resource)]
pub struct VisResource {
    pub swapl: Arc<Swapl>,
    pub lidar: Arc<Mutex<Lidar>>,
}