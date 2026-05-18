#!/usr/bin/env python3
"""
Virtual ROS interface — TCPROS 二进制序列化，零依赖，无需 ROS Master。

与 Rust `ros_bridge::messages` 完全对应（保持位兼容）：

消息类型:
  - RosMarkerArray  ← Perple 发的检测/跟踪结果
  - RosTwistStamped ← Perple 发的自车速度
  - PointCloud2     ← Perple 收的输入点云（Python 侧编码后发送）

用法:
  # 解码 Perple 发布的 MarkerArray
  with open("markers.bin", "rb") as f:
      markers = decode_marker_array(f)
      print(markers)

  # 编码 PointCloud2 发送给 Perple
  points = np.random.randn(1000, 3).astype(np.float32)
  frame = encode_pointcloud2(points, frame_id="lidar")
  sock.send(frame)
"""

import io
import socket
import struct
from dataclasses import dataclass, field
from typing import IO, List, Optional, Tuple

# ═════════════════════════════════════════════
#  ROS 标准消息类型（与 messages.rs 一致）
# ═════════════════════════════════════════════


@dataclass
class RosHeader:
    seq: int = 0
    sec: int = 0
    nsec: int = 0
    frame_id: str = ""


@dataclass
class RosPoint:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class RosVector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class RosQuaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class RosPose:
    position: RosPoint = field(default_factory=RosPoint)
    orientation: RosQuaternion = field(default_factory=RosQuaternion)


@dataclass
class RosTwist:
    linear: RosVector3 = field(default_factory=RosVector3)
    angular: RosVector3 = field(default_factory=RosVector3)


@dataclass
class RosTwistStamped:
    header: RosHeader = field(default_factory=RosHeader)
    twist: RosTwist = field(default_factory=RosTwist)


@dataclass
class RosColorRGBA:
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0


@dataclass
class RosMarker:
    header: RosHeader = field(default_factory=RosHeader)
    ns: str = ""
    id: int = 0
    type_: int = 1  # 1 = CUBE
    action: int = 0  # 0 = ADD/MODIFY
    pose: RosPose = field(default_factory=RosPose)
    scale: RosVector3 = field(default_factory=lambda: RosVector3(1.0, 1.0, 1.0))
    color: RosColorRGBA = field(default_factory=lambda: RosColorRGBA(1.0, 1.0, 1.0, 1.0))
    lifetime_sec: int = 0
    lifetime_nsec: int = 0
    frame_locked: bool = False
    points: List[RosPoint] = field(default_factory=list)
    colors: List[RosColorRGBA] = field(default_factory=list)
    text: str = ""
    mesh_resource: str = ""
    mesh_use_embedded_materials: bool = False


@dataclass
class RosMarkerArray:
    markers: List[RosMarker] = field(default_factory=list)


# ═════════════════════════════════════════════
#  TCPROS 二进制编码/解码
# ═════════════════════════════════════════════

def _write_f64(w: IO[bytes], v: float) -> None:
    w.write(struct.pack("<d", v))


def _read_f64(r: IO[bytes]) -> float:
    return struct.unpack("<d", r.read(8))[0]


def _write_f32(w: IO[bytes], v: float) -> None:
    w.write(struct.pack("<f", v))


def _read_f32(r: IO[bytes]) -> float:
    return struct.unpack("<f", r.read(4))[0]


def _write_u32(w: IO[bytes], v: int) -> None:
    w.write(struct.pack("<I", v))


def _read_u32(r: IO[bytes]) -> int:
    return struct.unpack("<I", r.read(4))[0]


def _write_i32(w: IO[bytes], v: int) -> None:
    w.write(struct.pack("<i", v))


def _read_i32(r: IO[bytes]) -> int:
    return struct.unpack("<i", r.read(4))[0]


def _write_u8(w: IO[bytes], v: int) -> None:
    w.write(struct.pack("<B", v))


def _read_u8(r: IO[bytes]) -> int:
    return struct.unpack("<B", r.read(1))[0]


def _write_string(w: IO[bytes], s: str) -> None:
    raw = s.encode("utf-8")
    _write_u32(w, len(raw))
    w.write(raw)


def _read_string(r: IO[bytes]) -> str:
    length = _read_u32(r)
    return r.read(length).decode("utf-8")


# ── Header ──


def encode_header(w: IO[bytes], msg: RosHeader) -> None:
    _write_u32(w, msg.seq)
    _write_u32(w, msg.sec)
    _write_u32(w, msg.nsec)
    _write_string(w, msg.frame_id)


def decode_header(r: IO[bytes]) -> RosHeader:
    return RosHeader(
        seq=_read_u32(r),
        sec=_read_u32(r),
        nsec=_read_u32(r),
        frame_id=_read_string(r),
    )


# ── Point ──


def encode_point(w: IO[bytes], msg: RosPoint) -> None:
    _write_f64(w, msg.x)
    _write_f64(w, msg.y)
    _write_f64(w, msg.z)


def decode_point(r: IO[bytes]) -> RosPoint:
    return RosPoint(x=_read_f64(r), y=_read_f64(r), z=_read_f64(r))


# ── Vector3 ──


def encode_vector3(w: IO[bytes], msg: RosVector3) -> None:
    _write_f64(w, msg.x)
    _write_f64(w, msg.y)
    _write_f64(w, msg.z)


def decode_vector3(r: IO[bytes]) -> RosVector3:
    return RosVector3(x=_read_f64(r), y=_read_f64(r), z=_read_f64(r))


# ── Quaternion ──


def encode_quaternion(w: IO[bytes], msg: RosQuaternion) -> None:
    _write_f64(w, msg.x)
    _write_f64(w, msg.y)
    _write_f64(w, msg.z)
    _write_f64(w, msg.w)


def decode_quaternion(r: IO[bytes]) -> RosQuaternion:
    return RosQuaternion(
        x=_read_f64(r), y=_read_f64(r), z=_read_f64(r), w=_read_f64(r)
    )


# ── Pose ──


def encode_pose(w: IO[bytes], msg: RosPose) -> None:
    encode_point(w, msg.position)
    encode_quaternion(w, msg.orientation)


def decode_pose(r: IO[bytes]) -> RosPose:
    return RosPose(position=decode_point(r), orientation=decode_quaternion(r))


# ── Twist ──


def encode_twist(w: IO[bytes], msg: RosTwist) -> None:
    encode_vector3(w, msg.linear)
    encode_vector3(w, msg.angular)


def decode_twist(r: IO[bytes]) -> RosTwist:
    return RosTwist(linear=decode_vector3(r), angular=decode_vector3(r))


# ── TwistStamped ──


def encode_twist_stamped(w: IO[bytes], msg: RosTwistStamped) -> None:
    encode_header(w, msg.header)
    encode_twist(w, msg.twist)


def decode_twist_stamped(r: IO[bytes]) -> RosTwistStamped:
    return RosTwistStamped(header=decode_header(r), twist=decode_twist(r))


# ── ColorRGBA ──


def encode_color_rgba(w: IO[bytes], msg: RosColorRGBA) -> None:
    _write_f32(w, msg.r)
    _write_f32(w, msg.g)
    _write_f32(w, msg.b)
    _write_f32(w, msg.a)


def decode_color_rgba(r: IO[bytes]) -> RosColorRGBA:
    return RosColorRGBA(
        r=_read_f32(r), g=_read_f32(r), b=_read_f32(r), a=_read_f32(r)
    )


# ── Marker ──


def encode_marker(w: IO[bytes], msg: RosMarker) -> None:
    encode_header(w, msg.header)
    _write_string(w, msg.ns)
    _write_i32(w, msg.id)
    _write_i32(w, msg.type_)
    _write_i32(w, msg.action)
    encode_pose(w, msg.pose)
    encode_vector3(w, msg.scale)
    encode_color_rgba(w, msg.color)
    _write_u32(w, msg.lifetime_sec)
    _write_u32(w, msg.lifetime_nsec)
    _write_u8(w, 1 if msg.frame_locked else 0)
    _write_u32(w, len(msg.points))
    for p in msg.points:
        encode_point(w, p)
    _write_u32(w, len(msg.colors))
    for c in msg.colors:
        encode_color_rgba(w, c)
    _write_string(w, msg.text)
    _write_string(w, msg.mesh_resource)
    _write_u8(w, 1 if msg.mesh_use_embedded_materials else 0)


def decode_marker(r: IO[bytes]) -> RosMarker:
    header = decode_header(r)
    ns = _read_string(r)
    id = _read_i32(r)
    type_ = _read_i32(r)
    action = _read_i32(r)
    pose = decode_pose(r)
    scale = decode_vector3(r)
    color = decode_color_rgba(r)
    lifetime_sec = _read_u32(r)
    lifetime_nsec = _read_u32(r)
    frame_locked = _read_u8(r) != 0
    n_pts = _read_u32(r)
    points = [decode_point(r) for _ in range(n_pts)]
    n_cols = _read_u32(r)
    colors = [decode_color_rgba(r) for _ in range(n_cols)]
    text = _read_string(r)
    mesh_resource = _read_string(r)
    mesh_use = _read_u8(r) != 0
    return RosMarker(
        header=header, ns=ns, id=id, type_=type_, action=action,
        pose=pose, scale=scale, color=color,
        lifetime_sec=lifetime_sec, lifetime_nsec=lifetime_nsec,
        frame_locked=frame_locked, points=points, colors=colors,
        text=text, mesh_resource=mesh_resource,
        mesh_use_embedded_materials=mesh_use,
    )


# ── MarkerArray ──


def encode_marker_array(w: IO[bytes], msg: RosMarkerArray) -> None:
    _write_u32(w, len(msg.markers))
    for m in msg.markers:
        encode_marker(w, m)


def decode_marker_array(r: IO[bytes]) -> RosMarkerArray:
    n = _read_u32(r)
    markers = [decode_marker(r) for _ in range(n)]
    return RosMarkerArray(markers=markers)


# ═════════════════════════════════════════════
#  TCPROS 帧打包工具
# ═════════════════════════════════════════════


def tcpros_frame(payload: bytes) -> bytes:
    """给 payload 加上 4 字节长度前缀（TCPROS 线格式）。"""
    return struct.pack("<I", len(payload)) + payload


# ═════════════════════════════════════════════
#  PointCloud2 编码（输入侧）
# ═════════════════════════════════════════════

# PointField 定义
POINT_FIELD_X = b"\x01\x00\x00\x00"  # name="x"(len=1), offset=0, type=FLOAT32(7), count=1
POINT_FIELD_Y = b"\x01\x00\x00\x00"  # name="y"(len=1), offset=4, type=FLOAT32(7), count=1
POINT_FIELD_Z = b"\x01\x00\x00\x00"  # name="z"(len=1), offset=8, type=FLOAT32(7), count=1


def encode_pointcloud2(
    points: "np.ndarray",
    frame_id: str = "lidar",
    seq: int = 0,
    sec: int = 0,
    nsec: int = 0,
) -> bytes:
    """
    将 N×3 float32 点云编码为 TCPROS 帧格式的 PointCloud2。

    参数:
        points: numpy (N, 3) float32 数组
        frame_id: 坐标系

    返回:
        bytes: 完整的 TCPROS 帧（4 字节长度前缀 + payload）
    """
    import numpy as np

    n = len(points)
    point_step = 16  # 3×float32 + 1 padding
    row_step = n * point_step
    data = np.zeros((n, 4), dtype=np.float32)
    data[:, :3] = points
    data_bytes = data.tobytes()  # 16 bytes per point

    buf = io.BytesIO()
    # header
    _write_u32(buf, seq)
    _write_u32(buf, sec)
    _write_u32(buf, nsec)
    _write_string(buf, frame_id)
    # height, width
    _write_u32(buf, 1)  # height=1 (unorganized)
    _write_u32(buf, n)  # width=N
    # fields (3 PointFields)
    _write_u32(buf, 3)
    # field 0: x
    _write_string(buf, "x")
    _write_u32(buf, 0)  # offset
    _write_u8(buf, 7)  # FLOAT32
    _write_u32(buf, 1)  # count
    # field 1: y
    _write_string(buf, "y")
    _write_u32(buf, 4)  # offset
    _write_u8(buf, 7)  # FLOAT32
    _write_u32(buf, 1)  # count
    # field 2: z
    _write_string(buf, "z")
    _write_u32(buf, 8)  # offset
    _write_u8(buf, 7)  # FLOAT32
    _write_u32(buf, 1)  # count
    # is_bigendian
    _write_u8(buf, 0)
    # point_step, row_step
    _write_u32(buf, point_step)
    _write_u32(buf, row_step)
    # data
    _write_u32(buf, len(data_bytes))
    buf.write(data_bytes)
    # is_dense
    _write_u8(buf, 1)

    payload = buf.getvalue()
    return tcpros_frame(payload)


# ═════════════════════════════════════════════
#  TCP 客户端（向 Perple 发点云 + 收可视化结果）
# ═════════════════════════════════════════════


class VirtualRosClient:
    """通过 TCP 直连 Perple 的虚拟 ROS 接口（无需 ROS Master）。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 9090):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"[虚拟ROS] 已连接 {self.host}:{self.port}")

    def close(self) -> None:
        if self.sock:
            self.sock.close()
            self.sock = None

    def send_pointcloud(
        self,
        points: "np.ndarray",
        frame_id: str = "lidar",
    ) -> None:
        """发送 PointCloud2 到 Perple。"""
        if self.sock is None:
            raise RuntimeError("未连接")
        frame = encode_pointcloud2(points, frame_id=frame_id)
        self.sock.sendall(frame)
        print(f"[虚拟ROS] 已发送 {len(points)} 个点")

    def recv_marker_array(self) -> Optional[RosMarkerArray]:
        """接收一个 MarkerArray TCPROS 帧。"""
        if self.sock is None:
            raise RuntimeError("未连接")
        # 读 4 字节长度前缀
        raw_len = self.sock.recv(4)
        if not raw_len or len(raw_len) < 4:
            return None
        payload_len = struct.unpack("<I", raw_len)[0]
        payload = b""
        while len(payload) < payload_len:
            chunk = self.sock.recv(payload_len - len(payload))
            if not chunk:
                return None
            payload += chunk
        return decode_marker_array(io.BytesIO(payload))

    def recv_twist_stamped(self) -> Optional[RosTwistStamped]:
        """接收一个 TwistStamped TCPROS 帧。"""
        if self.sock is None:
            raise RuntimeError("未连接")
        raw_len = self.sock.recv(4)
        if not raw_len or len(raw_len) < 4:
            return None
        payload_len = struct.unpack("<I", raw_len)[0]
        payload = b""
        while len(payload) < payload_len:
            chunk = self.sock.recv(payload_len - len(payload))
            if not chunk:
                return None
            payload += chunk
        return decode_twist_stamped(io.BytesIO(payload))


# ═════════════════════════════════════════════
#  TCP 服务器（接收 Perple 发布的数据）
# ═════════════════════════════════════════════


class VirtualRosServer:
    """
    虚拟 ROS 服务器，接收 Perple 通过 ros_bridge 发布的数据。

    工作模式:
      1. 建立原始 TCP 连接（Perple 侧需改为直接 TCP 发布）
      2. 读取 TCPROS 帧并解码为 Python 对象
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9090):
        self.host = host
        self.port = port
        self.server: Optional[socket.socket] = None
        self._running = False

    def start(self) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self._running = True
        print(f"[虚拟ROS] 服务器启动于 {self.host}:{self.port}")

    def stop(self) -> None:
        self._running = False
        if self.server:
            self.server.close()
            self.server = None

    def serve_once(self, timeout: float = 10.0) -> None:
        """接受一个连接并持续读取数据。"""
        if self.server is None:
            raise RuntimeError("服务器未启动")
        self.server.settimeout(timeout)
        try:
            conn, addr = self.server.accept()
            print(f"[虚拟ROS] 客户端连接: {addr}")
            with conn:
                conn.settimeout(None)
                buf = b""
                while self._running:
                    try:
                        data = conn.recv(65536)
                        if not data:
                            break
                        buf += data
                        while len(buf) >= 4:
                            payload_len = struct.unpack("<I", buf[:4])[0]
                            if len(buf) < 4 + payload_len:
                                break
                            payload = buf[4:4 + payload_len]
                            buf = buf[4 + payload_len:]
                            self._on_frame(payload)
                    except socket.timeout:
                        continue
        except socket.timeout:
            pass

    def _on_frame(self, payload: bytes) -> None:
        """尝试解码收到的帧。"""
        r = io.BytesIO(payload)
        # 尝试按 MarkerArray 解码
        try:
            markers = decode_marker_array(r)
            print(f"[虚拟ROS] 收到 MarkerArray: {len(markers.markers)} 个标记")
            return
        except Exception:
            pass

        r.seek(0)
        try:
            twist = decode_twist_stamped(r)
            print(f"[虚拟ROS] 收到 TwistStamped: linear=({twist.twist.linear.x:.2f}, "
                  f"{twist.twist.linear.y:.2f}, {twist.twist.linear.z:.2f})")
            return
        except Exception:
            pass

        print(f"[虚拟ROS] 收到 {len(payload)} 字节，无法识别的消息类型")


# ═════════════════════════════════════════════
#  CLI 入口
# ═════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Perple 虚拟 ROS 接口")
    parser.add_argument("mode", choices=["server", "send"], help="运行模式")
    parser.add_argument("--host", default="127.0.0.1", help="主机地址")
    parser.add_argument("--port", type=int, default=9090, help="端口号")
    parser.add_argument("--num-points", type=int, default=5000, help="发送的点数 (send 模式)")
    parser.add_argument("--frame-id", default="lidar", help="坐标系名称")
    args = parser.parse_args()

    if args.mode == "server":
        server = VirtualRosServer(host=args.host, port=args.port)
        server.start()
        print("按 Ctrl+C 停止...")
        try:
            while True:
                server.serve_once()
        except KeyboardInterrupt:
            print("\n正在停止...")
        finally:
            server.stop()
    elif args.mode == "send":
        import numpy as np

        client = VirtualRosClient(host=args.host, port=args.port)
        try:
            client.connect()
            points = np.random.randn(args.num_points, 3).astype(np.float32)
            client.send_pointcloud(points, frame_id=args.frame_id)
            print("等待回复...")
            markers = client.recv_marker_array()
            if markers:
                print(f"收到 {len(markers.markers)} 个标记")
            else:
                print("无回复")
        finally:
            client.close()


if __name__ == "__main__":
    main()
