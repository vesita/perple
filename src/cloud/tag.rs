pub struct Tag3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rx: f32,
    pub ry: f32,
    pub rz: f32,
    pub xl: f32,
    pub yl: f32,
    pub zl: f32,
}

impl Tag3D {
    pub fn new(x: f32, y: f32, z: f32, rx: f32, ry: f32, rz: f32, xl: f32, yl: f32, zl: f32) -> Self {
        Tag3D {
            x,
            y,
            z,
            rx,
            ry,
            rz,
            xl,
            yl,
            zl,
        }
    }

    pub fn contains(&self, point: [f32; 3]) -> bool {
        let dx = point[0] - self.x;
        let dy = point[1] - self.y;
        let dz = point[2] - self.z;
        dx.abs() <= self.xl / 2.0 &&
        dy.abs() <= self.yl / 2.0 &&
        dz.abs() <= self.zl / 2.0
    }

    pub fn expand(&mut self, point: [f32; 3]) {
        let x_min = point[0].min(self.x - self.xl / 2.0);
        let y_min = point[1].min(self.y - self.yl / 2.0);
        let z_min = point[2].min(self.z - self.zl / 2.0);
        let x_max = point[0].max(self.x + self.xl / 2.0);
        let y_max = point[1].max(self.y + self.yl / 2.0);
        let z_max = point[2].max(self.z + self.zl / 2.0);
        self.xl = x_max - x_min;
        self.yl = y_max - y_min;
        self.zl = z_max - z_min;
        self.x = x_min + self.xl / 2.0;
        self.y = y_min + self.yl / 2.0;
        self.z = z_min + self.zl / 2.0;
    }

    pub fn empty_box() -> Self {
        Tag3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            rx: 0.0,
            ry: 0.0,
            rz: 0.0,
            xl: 0.0,
            yl: 0.0,
            zl: 0.0,
        }
    }

    pub fn cloud2box(&mut self, cloud3d: &Vec<[f32; 3]>) {
        if cloud3d.is_empty() {
            return;
        }
        
        // 计算点云在各轴上的极值
        let mut x_min = cloud3d[0][0];
        let mut x_max = cloud3d[0][0];
        let mut y_min = cloud3d[0][1];
        let mut y_max = cloud3d[0][1];
        let mut z_min = cloud3d[0][2];
        let mut z_max = cloud3d[0][2];
        
        for point in cloud3d {
            x_min = x_min.min(point[0]);
            x_max = x_max.max(point[0]);
            y_min = y_min.min(point[1]);
            y_max = y_max.max(point[1]);
            z_min = z_min.min(point[2]);
            z_max = z_max.max(point[2]);
        }
        
        // 计算中心点坐标
        let x = (x_min + x_max) / 2.0;
        let y = (y_min + y_max) / 2.0;
        let z = (z_min + z_max) / 2.0;
        
        // 计算各轴长度
        let xl = x_max - x_min;
        let yl = y_max - y_min;
        let zl = z_max - z_min;
        
        // 在没有rx和ry的情况下，创建一个合适的3D边界框
        // rz通常表示绕z轴的旋转，在这里设为0
        self.x = x;
        self.y = y;
        self.z = z;
        self.rx = 0.0;
        self.ry = 0.0;
        self.rz = 0.0;
        self.xl = xl;
        self.yl = yl;
        self.zl = zl;
    }

    pub fn look_down(&mut self, cloud2d: &Vec<[f32; 2]>) { 
        if cloud2d.is_empty() {
            return;
        }
        let mut x_min = cloud2d[0][0];
        let mut x_max = cloud2d[0][0];
        let mut y_min = cloud2d[0][1];
        let mut y_max = cloud2d[0][1];
        for point in cloud2d {
            x_min = x_min.min(point[0]);
            x_max = x_max.max(point[0]);
            y_min = y_min.min(point[1]);
            y_max = y_max.max(point[1]);
        }
        let x = (x_min + x_max) / 2.0;
        let y = (y_min + y_max) / 2.0;
        let xl = x_max - x_min;
        let yl = y_max - y_min;
        self.x = x;
        self.y = y;
        self.xl = xl;
        self.yl = yl;
        self.z = 0.0;
        self.zl = f32::MAX;
        self.rx = 0.0;
        self.ry = 0.0;
        self.rz = 0.0;
    }
}