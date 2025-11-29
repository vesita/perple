use std::sync::{Arc, Mutex};
use bevy::prelude::*;

use crate::{Swapl};

#[derive(Resource)]
pub struct VisResource {
    pub swapl: Arc<Mutex<Swapl>>,
}