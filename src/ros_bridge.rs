//! ROS1 桥接模块
//!
//! 将 Perple 内部数据通过 ROS1 话题发布/订阅。
//! 数据流：ROS topic → Swapl → Perple pipeline → Swapl → ROS topic

mod messages;

pub use messages::{
    RosHeader, RosPoint, RosVector3, RosQuaternion, RosPose,
    RosTwist, RosTwistStamped, RosColorRGBA, RosMarker, RosMarkerArray,
    marker_array_to_tcpros, twist_stamped_to_tcpros,
};

use std::io::{Cursor, Read};

use byteorder::{LittleEndian, ReadBytesExt};
use rosrust::{self, RawMessage};

use self::messages::{decode_header, decode_string};

#[cfg(feature = "ros1")]
use crate::cloud::ego_motion::EgoMotion;
#[cfg(feature = "ros1")]
use crate::cloud::CldBud;
#[cfg(feature = "ros1")]
use crate::swapl::global_swapl;
#[cfg(feature = "ros1")]
use crate::tracker::output::Target;

// ═════════════════════════════════════════════
//  转换：Perple 类型 → ROS 消息类型
// ═════════════════════════════════════════════

/// 将 Perple Target 转换为 ROS MarkerArray
fn targets_to_marker_array(
    targets: &[Target],
    frame_id: &str,
    stamp: rosrust::Time,
    seq: u32,
) -> RosMarkerArray {
    let header = RosHeader {
        seq,
        stamp,
        frame_id: frame_id.to_string(),
    };

    let markers: Vec<RosMarker> = targets
        .iter()
        .filter(|t| t.classification != "ground")
        .enumerate()
        .map(|(i, t)| target_to_marker(t, i as i32, &header))
        .collect();

    RosMarkerArray { markers }
}

fn target_to_marker(target: &Target, id: i32, header: &RosHeader) -> RosMarker {
    let center = target.the_box.center();

    // 颜色映射：遵循 PLAN.md 配色
    let (r, g, b) = if target.class_type == "person" {
        (0.0, 1.0, 1.0) // cyan
    } else {
        match target.classification.as_str() {
            "moving" => (1.0, 0.0, 0.0), // red
            "static" => (0.0, 1.0, 0.0), // green
            "movable" => (1.0, 1.0, 0.0), // yellow
            "floating" => (0.5, 0.5, 1.0), // light blue
            _ => (1.0, 1.0, 1.0), // white
        }
    };

    let mut marker = RosMarker {
        header: header.clone(),
        ns: "perple".to_string(),
        id,
        type_: 1, // CUBE
        action: 0, // ADD/MODIFY
        pose: RosPose {
            position: RosPoint {
                x: center.x as f64,
                y: center.y as f64,
                z: center.z as f64,
            },
            orientation: RosQuaternion {
                x: 0.0, y: 0.0, z: 0.0, w: 1.0,
            },
        },
        scale: RosVector3 {
            x: target.the_box.length as f64,
            y: target.the_box.width as f64,
            z: target.the_box.height as f64,
        },
        color: RosColorRGBA { r, g, b, a: 0.8 },
        text: format!("{} | {} | {:.1}m/s", target.id, target.classification, target.speed),
        ..Default::default()
    };

    if target.speed > 0.5 {
        // 用箭头表示高速目标
        marker.type_ = 0; // ARROW
        marker.points = vec![
            RosPoint { x: 0.0, y: 0.0, z: 0.0 },
            RosPoint {
                x: target.velocity[0] as f64,
                y: target.velocity[1] as f64,
                z: target.velocity[2] as f64,
            },
        ];
    }

    marker
}

/// 将 Perple CldBud 列表转换为 ROS MarkerArray（3D 检测框）
fn cldbuds_to_marker_array(buds: &[CldBud], frame_id: &str, stamp: rosrust::Time, seq: u32) -> RosMarkerArray {
    let header = RosHeader {
        seq,
        stamp,
        frame_id: frame_id.to_string(),
    };

    let markers: Vec<RosMarker> = buds
        .iter()
        .filter(|b| b.class_name != "ground" && b.class_name != "ceiling")
        .enumerate()
        .map(|(i, bud)| {
            let center = bud.the_box.center();
            RosMarker {
                header: header.clone(),
                ns: "perple_detections".to_string(),
                id: i as i32,
                type_: 1, // CUBE
                action: 0,
                pose: RosPose {
                    position: RosPoint {
                        x: center.x as f64,
                        y: center.y as f64,
                        z: center.z as f64,
                    },
                    orientation: RosQuaternion { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
                },
                scale: RosVector3 {
                    x: bud.the_box.length as f64,
                    y: bud.the_box.width as f64,
                    z: bud.the_box.height as f64,
                },
                color: RosColorRGBA { r: 0.5, g: 0.5, b: 1.0, a: 0.6 },
                text: format!("{} {:.0}%", bud.class_name, bud.confidence * 100.0),
                ..Default::default()
            }
        })
        .collect();

    RosMarkerArray { markers }
}

// ═════════════════════════════════════════════
//  ROS 桥接配置与主结构体
// ═════════════════════════════════════════════

/// ROS1 桥接配置
#[derive(Debug, Clone)]
pub struct RosBridgeConfig {
    pub node_name: String,
    pub input_cloud_topic: String,
    pub output_targets_topic: String,
    pub output_detections_topic: String,
    pub output_ego_velocity_topic: String,
    pub frame_id: String,
    pub publish_rate_hz: f64,
    pub queue_size: usize,
}

impl Default for RosBridgeConfig {
    fn default() -> Self {
        Self {
            node_name: "perple".to_string(),
            input_cloud_topic: "/perple/input/cloud".to_string(),
            output_targets_topic: "/perple/output/targets".to_string(),
            output_detections_topic: "/perple/output/detections".to_string(),
            output_ego_velocity_topic: "/perple/output/ego_velocity".to_string(),
            frame_id: "lidar".to_string(),
            publish_rate_hz: 20.0,
            queue_size: 10,
        }
    }
}

/// ROS1 桥接器
///
/// 通过 Swapl 与 Perple 管道通信：
/// - 输入：订阅 ROS 话题 → 写入 Swapl.clouds
/// - 输出：读取 Swapl.targets / Swapl.cld_objs → 发布 ROS 话题
#[cfg(feature = "ros1")]
pub struct RosBridge {
    config: RosBridgeConfig,
    ego_motion: EgoMotion,
    seq: u32,
    targets_pub: Option<rosrust::Publisher<RawMessage>>,
    detections_pub: Option<rosrust::Publisher<RawMessage>>,
    ego_vel_pub: Option<rosrust::Publisher<RawMessage>>,
    _cloud_sub: Option<rosrust::Subscriber>,
}

#[cfg(feature = "ros1")]
impl RosBridge {
    pub fn new(config: RosBridgeConfig) -> Self {
        Self {
            config,
            ego_motion: EgoMotion::new(),
            seq: 0,
            targets_pub: None,
            detections_pub: None,
            ego_vel_pub: None,
            _cloud_sub: None,
        }
    }

    /// 初始化 ROS 节点（需要在 tokio runtime 外/前调用）
    pub fn init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        rosrust::init(&self.config.node_name);
        log::info!("ROS 节点初始化完成: {}", self.config.node_name);

        // 创建发布器
        self.targets_pub = Some(rosrust::publish(
            &self.config.output_targets_topic,
            self.config.queue_size,
        )?);
        self.detections_pub = Some(rosrust::publish(
            &self.config.output_detections_topic,
            self.config.queue_size,
        )?);
        self.ego_vel_pub = Some(rosrust::publish(
            &self.config.output_ego_velocity_topic,
            self.config.queue_size,
        )?);
        log::info!("ROS 话题发布器已创建");

        // 创建订阅器：输入点云
        let cloud_topic = self.config.input_cloud_topic.clone();
        self._cloud_sub = Some(rosrust::subscribe::<RawMessage, _>(
            &cloud_topic,
            self.config.queue_size,
            move |msg: RawMessage| {
                Self::on_input_cloud(msg.0);
            },
        )?);
        log::info!("已订阅: {}", cloud_topic);

        Ok(())
    }

    /// 发布所有输出话题（每帧调用一次）
    pub fn publish_all(&mut self) {
        self.seq += 1;
        let stamp = rosrust::now();
        let frame_id = &self.config.frame_id;

        // ── 自车速度估计 ──
        let ego_vel = self.ego_motion.update();

        // ── 发布跟踪目标 (MarkerArray) ──
        let swapl = global_swapl();
        if let Some(ref pub_) = self.targets_pub {
            if let Some(targets) = swapl.targets.lock().unwrap().peek_latest() {
                let markers = targets_to_marker_array(&targets, frame_id, stamp, self.seq);
                let bytes = marker_array_to_tcpros(&markers);
                pub_.send(RawMessage(bytes)).ok();
            }
        }

        // ── 发布 3D 检测框 (MarkerArray) ──
        if let Some(ref pub_) = self.detections_pub {
            if let Some(buds) = swapl.cld_objs.lock().unwrap().peek_latest() {
                let markers = cldbuds_to_marker_array(&buds, frame_id, stamp, self.seq);
                let bytes = marker_array_to_tcpros(&markers);
                pub_.send(RawMessage(bytes)).ok();
            }
        }

        // ── 发布自车速度 (TwistStamped) ──
        if let Some(ref pub_) = self.ego_vel_pub {
            let twist = RosTwistStamped {
                header: RosHeader {
                    seq: self.seq,
                    stamp,
                    frame_id: frame_id.to_string(),
                },
                twist: RosTwist {
                    linear: RosVector3 {
                        x: ego_vel[0] as f64,
                        y: ego_vel[1] as f64,
                        z: ego_vel[2] as f64,
                    },
                    angular: RosVector3 { x: 0.0, y: 0.0, z: 0.0 },
                },
            };
            let bytes = twist_stamped_to_tcpros(&twist);
            pub_.send(RawMessage(bytes)).ok();
        }
    }

    /// 订阅回调：输入点云
    fn on_input_cloud(data: Vec<u8>) {
        // 跳过 TCPROS 4 字节长度前缀
        if data.len() < 4 {
            return;
        }
        let payload_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < 4 + payload_len {
            return;
        }

        let mut cursor = Cursor::new(&data[4..4 + payload_len]);

        // 解析 PointCloud2: 跳过 header 和 fields，提取 XYZ
        // 简化实现：解析点云结构并提取 [x, y, z] 点
        if let Some(points) = parse_pointcloud2_payload(&mut cursor) {
            let swapl = global_swapl();
            let mut stream = swapl.clouds.lock().unwrap();
            let _ = stream.write(points);
        }
    }
}

// ═════════════════════════════════════════════
//  PointCloud2 解析（最小实现）
// ═════════════════════════════════════════════

/// 从 PointCloud2 中提取 XYZ 点（跳过 4 字节长度前缀后的 payload）
fn parse_pointcloud2_payload<R: Read>(r: &mut R) -> Option<Vec<[f32; 3]>> {
    // PointCloud2: Header header, uint32 height, uint32 width,
    // PointField[] fields, bool is_bigendian, uint32 point_step,
    // uint32 row_step, uint8[] data, bool is_dense
    //
    // 我们只关心: width * height 个点, point_step 字节/点
    // 跳过 header (seq + time + frame_id = 4+4+4+4+len(string))
    if let Ok(header) = decode_header(r) {
        // skip header already consumed
        let header_msg = header;
        let _ = header_msg;
    } else {
        return None;
    }

    let height = r.read_u32::<LittleEndian>().ok()?;
    let width = r.read_u32::<LittleEndian>().ok()?;

    // 跳过 fields
    let n_fields = r.read_u32::<LittleEndian>().ok()? as usize;
    for _ in 0..n_fields {
        // name(string), offset(uint32), datatype(uint8), count(uint32)
        decode_string(r).ok()?;
        r.read_u32::<LittleEndian>().ok()?;
        r.read_u8().ok()?;
        r.read_u32::<LittleEndian>().ok()?;
    }

    let is_bigendian = r.read_u8().ok()?;
    let _point_step = r.read_u32::<LittleEndian>().ok()?;
    let _row_step = r.read_u32::<LittleEndian>().ok()?;

    // data array
    let n_points = (width * height) as usize;
    let data_len = r.read_u32::<LittleEndian>().ok()? as usize;
    let mut data = vec![0u8; data_len];
    r.read_exact(&mut data).ok()?;

    let _is_dense = r.read_u8().ok();

    if is_bigendian != 0 {
        log::warn!("PointCloud2 is big-endian, skipping");
        return None;
    }

    // 从原始数据中提取 XYZ (FLOAT32 @ offset 0, 4, 8)
    let step = _point_step as usize;
    let mut points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let offset = i * step;
        if offset + 12 > data.len() {
            break;
        }
        let x = f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        let y = f32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        let z = f32::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]);
        if x.is_finite() && y.is_finite() && z.is_finite() {
            points.push([x, y, z]);
        }
    }

    log::info!("收到点云: {} 点 (解析 {} 点)", n_points, points.len());
    Some(points)
}

// ═════════════════════════════════════════════
//  非 ROS1 模式下的空实现
// ═════════════════════════════════════════════

#[cfg(not(feature = "ros1"))]
pub struct RosBridge;

#[cfg(not(feature = "ros1"))]
impl RosBridge {
    pub fn new(_config: RosBridgeConfig) -> Self {
        Self
    }
    pub fn init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Err("ros1 feature not enabled. Build with --features ros1".into())
    }
    pub fn publish_all(&mut self) {}
}
