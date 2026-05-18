//! ROS 标准消息类型定义 + 二进制序列化
//!
//! 严格遵循 ROS 二进制序列化格式（little-endian TCPROS）。

use std::io::{self, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

// ═════════════════════════════════════════════
//  ROS 标准消息类型定义
// ═════════════════════════════════════════════

/// std_msgs/Header
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RosHeader {
    pub seq: u32,
    pub stamp: rosrust::Time,
    pub frame_id: String,
}

/// geometry_msgs/Point
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RosPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// geometry_msgs/Vector3
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RosVector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// geometry_msgs/Quaternion
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RosQuaternion {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

/// geometry_msgs/Pose
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RosPose {
    pub position: RosPoint,
    pub orientation: RosQuaternion,
}

/// geometry_msgs/Twist
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RosTwist {
    pub linear: RosVector3,
    pub angular: RosVector3,
}

/// geometry_msgs/TwistStamped
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RosTwistStamped {
    pub header: RosHeader,
    pub twist: RosTwist,
}

/// std_msgs/ColorRGBA
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RosColorRGBA {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// visualization_msgs/Marker
#[derive(Debug, Clone)]
pub struct RosMarker {
    pub header: RosHeader,
    pub ns: String,
    pub id: i32,
    pub type_: i32,
    pub action: i32,
    pub pose: RosPose,
    pub scale: RosVector3,
    pub color: RosColorRGBA,
    pub lifetime: rosrust::Duration,
    pub frame_locked: bool,
    pub points: Vec<RosPoint>,
    pub colors: Vec<RosColorRGBA>,
    pub text: String,
    pub mesh_resource: String,
    pub mesh_use_embedded_materials: bool,
}

impl Default for RosMarker {
    fn default() -> Self {
        Self {
            header: RosHeader::default(),
            ns: String::new(),
            id: 0,
            type_: 1, // CUBE
            action: 0, // ADD/MODIFY
            pose: RosPose::default(),
            scale: RosVector3 { x: 1.0, y: 1.0, z: 1.0 },
            color: RosColorRGBA { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
            lifetime: rosrust::Duration::from_nanos(0),
            frame_locked: false,
            points: Vec::new(),
            colors: Vec::new(),
            text: String::new(),
            mesh_resource: String::new(),
            mesh_use_embedded_materials: false,
        }
    }
}

/// visualization_msgs/MarkerArray
#[derive(Debug, Clone, Default)]
pub struct RosMarkerArray {
    pub markers: Vec<RosMarker>,
}

// ═════════════════════════════════════════════
//  序列化：encode_* / decode_* 函数
// ═════════════════════════════════════════════

fn encode_header<W: Write>(msg: &RosHeader, w: &mut W) -> io::Result<()> {
    w.write_u32::<LittleEndian>(msg.seq)?;
    w.write_u32::<LittleEndian>(msg.stamp.sec as u32)?;
    w.write_u32::<LittleEndian>(msg.stamp.nsec as u32)?;
    encode_string(&msg.frame_id, w)?;
    Ok(())
}

pub fn decode_header<R: Read>(r: &mut R) -> io::Result<RosHeader> {
    let seq = r.read_u32::<LittleEndian>()?;
    let sec = r.read_u32::<LittleEndian>()?;
    let nsec = r.read_u32::<LittleEndian>()?;
    let frame_id = decode_string(r)?;
    Ok(RosHeader {
        seq,
        stamp: rosrust::Time { sec, nsec },
        frame_id,
    })
}

fn encode_point<W: Write>(msg: &RosPoint, w: &mut W) -> io::Result<()> {
    w.write_f64::<LittleEndian>(msg.x)?;
    w.write_f64::<LittleEndian>(msg.y)?;
    w.write_f64::<LittleEndian>(msg.z)?;
    Ok(())
}

fn decode_point<R: Read>(r: &mut R) -> io::Result<RosPoint> {
    Ok(RosPoint {
        x: r.read_f64::<LittleEndian>()?,
        y: r.read_f64::<LittleEndian>()?,
        z: r.read_f64::<LittleEndian>()?,
    })
}

fn encode_vector3<W: Write>(msg: &RosVector3, w: &mut W) -> io::Result<()> {
    w.write_f64::<LittleEndian>(msg.x)?;
    w.write_f64::<LittleEndian>(msg.y)?;
    w.write_f64::<LittleEndian>(msg.z)?;
    Ok(())
}

fn decode_vector3<R: Read>(r: &mut R) -> io::Result<RosVector3> {
    Ok(RosVector3 {
        x: r.read_f64::<LittleEndian>()?,
        y: r.read_f64::<LittleEndian>()?,
        z: r.read_f64::<LittleEndian>()?,
    })
}

fn encode_quaternion<W: Write>(msg: &RosQuaternion, w: &mut W) -> io::Result<()> {
    w.write_f64::<LittleEndian>(msg.x)?;
    w.write_f64::<LittleEndian>(msg.y)?;
    w.write_f64::<LittleEndian>(msg.z)?;
    w.write_f64::<LittleEndian>(msg.w)?;
    Ok(())
}

fn decode_quaternion<R: Read>(r: &mut R) -> io::Result<RosQuaternion> {
    Ok(RosQuaternion {
        x: r.read_f64::<LittleEndian>()?,
        y: r.read_f64::<LittleEndian>()?,
        z: r.read_f64::<LittleEndian>()?,
        w: r.read_f64::<LittleEndian>()?,
    })
}

fn encode_pose<W: Write>(msg: &RosPose, w: &mut W) -> io::Result<()> {
    encode_point(&msg.position, w)?;
    encode_quaternion(&msg.orientation, w)
}

fn decode_pose<R: Read>(r: &mut R) -> io::Result<RosPose> {
    Ok(RosPose {
        position: decode_point(r)?,
        orientation: decode_quaternion(r)?,
    })
}

fn encode_twist<W: Write>(msg: &RosTwist, w: &mut W) -> io::Result<()> {
    encode_vector3(&msg.linear, w)?;
    encode_vector3(&msg.angular, w)
}

fn decode_twist<R: Read>(r: &mut R) -> io::Result<RosTwist> {
    Ok(RosTwist {
        linear: decode_vector3(r)?,
        angular: decode_vector3(r)?,
    })
}

fn encode_twist_stamped<W: Write>(msg: &RosTwistStamped, w: &mut W) -> io::Result<()> {
    encode_header(&msg.header, w)?;
    encode_twist(&msg.twist, w)
}

fn decode_twist_stamped<R: Read>(r: &mut R) -> io::Result<RosTwistStamped> {
    Ok(RosTwistStamped {
        header: decode_header(r)?,
        twist: decode_twist(r)?,
    })
}

fn encode_color_rgba<W: Write>(msg: &RosColorRGBA, w: &mut W) -> io::Result<()> {
    w.write_f32::<LittleEndian>(msg.r)?;
    w.write_f32::<LittleEndian>(msg.g)?;
    w.write_f32::<LittleEndian>(msg.b)?;
    w.write_f32::<LittleEndian>(msg.a)?;
    Ok(())
}

fn decode_color_rgba<R: Read>(r: &mut R) -> io::Result<RosColorRGBA> {
    Ok(RosColorRGBA {
        r: r.read_f32::<LittleEndian>()?,
        g: r.read_f32::<LittleEndian>()?,
        b: r.read_f32::<LittleEndian>()?,
        a: r.read_f32::<LittleEndian>()?,
    })
}

fn encode_string<W: Write>(s: &str, w: &mut W) -> io::Result<()> {
    let bytes = s.as_bytes();
    w.write_u32::<LittleEndian>(bytes.len() as u32)?;
    w.write_all(bytes)
}

pub(crate) fn decode_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = r.read_u32::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn encode_duration<W: Write>(d: &rosrust::Duration, w: &mut W) -> io::Result<()> {
    w.write_u32::<LittleEndian>(d.sec as u32)?;
    w.write_u32::<LittleEndian>(d.nsec as u32)?;
    Ok(())
}

fn decode_duration<R: Read>(r: &mut R) -> io::Result<rosrust::Duration> {
    Ok(rosrust::Duration {
        sec: r.read_u32::<LittleEndian>()? as i32,
        nsec: r.read_u32::<LittleEndian>()? as i32,
    })
}

fn encode_marker<W: Write>(msg: &RosMarker, w: &mut W) -> io::Result<()> {
    encode_header(&msg.header, w)?;
    encode_string(&msg.ns, w)?;
    w.write_i32::<LittleEndian>(msg.id)?;
    w.write_i32::<LittleEndian>(msg.type_)?;
    w.write_i32::<LittleEndian>(msg.action)?;
    encode_pose(&msg.pose, w)?;
    encode_vector3(&msg.scale, w)?;
    encode_color_rgba(&msg.color, w)?;
    encode_duration(&msg.lifetime, w)?;
    w.write_u8(msg.frame_locked as u8)?;
    // points array
    w.write_u32::<LittleEndian>(msg.points.len() as u32)?;
    for p in &msg.points {
        encode_point(p, w)?;
    }
    // colors array
    w.write_u32::<LittleEndian>(msg.colors.len() as u32)?;
    for c in &msg.colors {
        encode_color_rgba(c, w)?;
    }
    encode_string(&msg.text, w)?;
    encode_string(&msg.mesh_resource, w)?;
    w.write_u8(msg.mesh_use_embedded_materials as u8)?;
    Ok(())
}

fn decode_marker<R: Read>(r: &mut R) -> io::Result<RosMarker> {
    let header = decode_header(r)?;
    let ns = decode_string(r)?;
    let id = r.read_i32::<LittleEndian>()?;
    let type_ = r.read_i32::<LittleEndian>()?;
    let action = r.read_i32::<LittleEndian>()?;
    let pose = decode_pose(r)?;
    let scale = decode_vector3(r)?;
    let color = decode_color_rgba(r)?;
    let lifetime = decode_duration(r)?;
    let frame_locked = r.read_u8()? > 0;
    // points array
    let n_points = r.read_u32::<LittleEndian>()? as usize;
    let mut points = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        points.push(decode_point(r)?);
    }
    // colors array
    let n_colors = r.read_u32::<LittleEndian>()? as usize;
    let mut colors = Vec::with_capacity(n_colors);
    for _ in 0..n_colors {
        colors.push(decode_color_rgba(r)?);
    }
    let text = decode_string(r)?;
    let mesh_resource = decode_string(r)?;
    let mesh_use_embedded_materials = r.read_u8()? > 0;
    Ok(RosMarker {
        header, ns, id, type_, action, pose, scale, color,
        lifetime, frame_locked, points, colors, text,
        mesh_resource, mesh_use_embedded_materials,
    })
}

fn encode_marker_array<W: Write>(msg: &RosMarkerArray, w: &mut W) -> io::Result<()> {
    w.write_u32::<LittleEndian>(msg.markers.len() as u32)?;
    for m in &msg.markers {
        encode_marker(m, w)?;
    }
    Ok(())
}

fn decode_marker_array<R: Read>(r: &mut R) -> io::Result<RosMarkerArray> {
    let n = r.read_u32::<LittleEndian>()? as usize;
    let mut markers = Vec::with_capacity(n);
    for _ in 0..n {
        markers.push(decode_marker(r)?);
    }
    Ok(RosMarkerArray { markers })
}

// ═════════════════════════════════════════════
//  TCPROS 打包：每帧消息以 4 字节长度前缀开头
// ═════════════════════════════════════════════

fn encode_tcpros_frame<F>(encoder: F) -> Vec<u8>
where
    F: Fn(&mut Vec<u8>) -> io::Result<()>,
{
    let mut buf = Vec::new();
    encoder(&mut buf).ok();
    let len = buf.len() as u32;
    let mut frame = Vec::with_capacity(4 + buf.len());
    frame.extend_from_slice(&len.to_le_bytes());
    frame.extend_from_slice(&buf);
    frame
}

// ═════════════════════════════════════════════
//  序列化辅助：消息 → TCPROS 帧
// ═════════════════════════════════════════════

pub fn marker_array_to_tcpros(msg: &RosMarkerArray) -> Vec<u8> {
    encode_tcpros_frame(|buf| encode_marker_array(msg, buf))
}

pub fn twist_stamped_to_tcpros(msg: &RosTwistStamped) -> Vec<u8> {
    encode_tcpros_frame(|buf| encode_twist_stamped(msg, buf))
}
