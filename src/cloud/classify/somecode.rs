


pub fn the_v(points_in_plane: &[[f32; 3];3], target: &[f32; 3]) -> f32 { 
    let v1 = [points_in_plane[1][0] - points_in_plane[0][0],
              points_in_plane[1][1] - points_in_plane[0][1],
              points_in_plane[1][2] - points_in_plane[0][2]];
    let v2 = [points_in_plane[2][0] - points_in_plane[0][0],
              points_in_plane[2][1] - points_in_plane[0][1],
              points_in_plane[2][2] - points_in_plane[0][2]];
    let v3 = [target[0] - points_in_plane[0][0],
              target[1] - points_in_plane[0][1],
              target[2] - points_in_plane[0][2]];
    
    v1[0] * v2[1] * v3[2] + v1[1] * v2[2] * v3[0] + v1[2] * v2[0] * v3[1] -
    v1[2] * v2[1] * v3[0] - v1[1] * v2[0] * v3[2] - v1[0] * v2[2] * v3[1]
}