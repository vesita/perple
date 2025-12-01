use std::sync::Arc;
use bevy::prelude::*;

use crate::{Swapl};

#[derive(Resource)]
pub struct VisResource {
    pub swapl: Arc<Swapl>,
}