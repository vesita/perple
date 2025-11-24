use crate::config::STREAM_CAPACITY;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::mem::MaybeUninit;
use std::boxed::Box;


/// 一个固定容量的线程安全流结构，用于在生产者和消费者之间传递数据
/// 推荐使用方法：
/// 获取写入位置的可变引用 -> 填充数据 -> 提交写入操作
/// 获取读取位置的引用 -> 处理数据 -> 提交读取操作
///

pub struct Stream<T: Default + Send + Clone> {
    pool: Vec<MaybeUninit<Option<T>>>,
    read_index: AtomicUsize,
    write_index: AtomicUsize,
}

/// 双向流结构，用于连接两个处理单元
/// 
/// IofActor 表示输入流的数据类型
/// OofActor 表示输出流的数据类型
pub struct Cream<IofActor: Default + Send + Clone, OofActor: Default + Send + Clone> {
    pub in_stream: Arc<Mutex<Stream<IofActor>>>,
    pub out_stream: Arc<Mutex<Stream<OofActor>>>,
}

impl<T: Default + Send + Clone> Stream<T> { 
    pub fn new() -> Self {
        // 创建一个预分配内存的Vec并在堆上分配
        let mut pool: Vec<MaybeUninit<Option<T>>> = Vec::with_capacity(STREAM_CAPACITY);
        
        // 初始化所有元素
        for _ in 0..STREAM_CAPACITY {
            pool.push(MaybeUninit::new(None));
        }
        
        Self {
            pool,
            read_index: AtomicUsize::new(0),
            write_index: AtomicUsize::new(0),
        }
    }
    
    /// 获取写入位置的可变引用，如果缓冲区已满则返回Err
    pub fn get_write_mut(&mut self) -> Result<&mut Option<T>, &'static str> {
        let current_read = self.read_index.load(Ordering::Acquire);
        let current_write = self.write_index.load(Ordering::Acquire);
        
        let next_index = (current_write + 1) % STREAM_CAPACITY;
        if next_index == current_read {
            return Err("缓冲区已满");
        }
        
        // 安全地获取可变引用
        Ok(unsafe { 
            self.pool[current_write].assume_init_mut()
        })
    }

        /// 获取读取位置的引用，如果缓冲区为空则返回None
    pub fn get_read_ref(&self) -> Option<&Option<T>> {
        let current_read = self.read_index.load(Ordering::Acquire);
        let current_write = self.write_index.load(Ordering::Acquire);
        
        if current_read == current_write {
            return None;
        }
        
        // 安全地获取不可变引用
        Some(unsafe {
            self.pool[current_read].assume_init_ref()
        })
    }
    
    /// 提交写入操作，将写索引向前移动
    pub fn commit_write(&mut self) -> Result<(), &'static str> {
        let current_read = self.read_index.load(Ordering::Acquire);
        let current_write = self.write_index.load(Ordering::Acquire);
        
        let next_index = (current_write + 1) % STREAM_CAPACITY;
        if next_index == current_read {
            return Err("缓冲区已满");
        }
        
        // 更新写索引
        self.write_index.store(next_index, Ordering::Release);
        Ok(())
    }
    

    
    /// 提交读取操作，将读索引向前移动
    pub fn commit_read(&mut self) -> Result<(), &'static str> {
        let current_read = self.read_index.load(Ordering::Acquire);
        let current_write = self.write_index.load(Ordering::Acquire);
        
        if current_read == current_write {
            return Err("缓冲区为空");
        }
        // 更新读索引
        self.read_index.store((current_read + 1) % STREAM_CAPACITY, Ordering::Release);
        Ok(())
    }
    
    /// 尝试写入数据
    pub fn write(&mut self, data: T) -> Result<(), &'static str> {
        let slot = self.get_write_mut()?;
        *slot = Some(data);
        self.commit_write()
    }
    
    /// 尝试读取数据
    pub fn read(&mut self) -> Option<T> {
        let data = self.get_read_ref()?.clone();
        self.commit_read().ok()?;
        data
    }

    /// 检查缓冲区是否有数据
    pub fn has_data(&self) -> bool {
        let current_read = self.read_index.load(Ordering::Acquire);
        let current_write = self.write_index.load(Ordering::Acquire);
        current_read != current_write
    }
    
    /// 直接写入到指定索引位置，无额外拷贝
    /// 通过读写标记保障数据一致性
    /// 直接写入到指定索引位置，无额外拷贝
    /// 通过读写标记保障数据一致性
    pub fn write_direct<F>(&mut self, writer: F) -> Result<(), &'static str>
    where
        F: FnOnce(&mut Option<T>),
    {
        let slot = self.get_write_mut()?;
        writer(slot);
        self.commit_write()
    }
    
}

impl<T: Default + Send + Clone> Stream<T> {
    // 克隆实现等其他方法...
}

impl <IofActor: Default + Send + Clone, OofActor: Default + Send + Clone> Cream<IofActor, OofActor> {
    pub fn new() -> Self {
        let in_stream = Arc::new(Mutex::new(Stream::<IofActor>::new()));
        let out_stream = Arc::new(Mutex::new(Stream::<OofActor>::new()));
        
        Self {
            in_stream,
            out_stream,
        }
    }

    /// 从输入流读取一个数据项并返回。这是主要的消费入口。
    pub fn read(&self) -> Option<IofActor> {
        self.in_stream.lock().unwrap().read()
    }
    
    /// 提交对输入流的读取操作，移动读取指针。
    pub fn commit_read(&self) {
        self.in_stream.lock().unwrap().commit_read().ok();
    }

    /// 向输出流写入一个数据项。
    pub fn write(&self, data: OofActor) -> Result<(), &'static str> {
        self.out_stream.lock().unwrap().write(data)
    }
    
    /// 提交对输出流的写入操作，移动写入指针。
    pub fn commit_write(&self) {
        self.out_stream.lock().unwrap().commit_write().ok();
    }

    /// 从输出流接收一个数据项。
    pub fn reciv(&self) -> Option<OofActor> {
        self.out_stream.lock().unwrap().read()
    }

    /// 提交对输出流的接收操作，移动接收指针。
    pub fn commit_reciv(&self) {
        self.out_stream.lock().unwrap().commit_read().ok();
    }

    /// 将一个数据项交付到输出流（等同于提交读取操作）。
    pub fn deliv(&self) -> Result<(), &'static str> {
        self.out_stream.lock().unwrap().commit_read()
    }

    /// 提交对输出流的交付操作，移动交付指针。
    pub fn commit_deliv(&self) {
        self.out_stream.lock().unwrap().commit_read().ok();
    }

    pub fn share_ioa(&self) -> Arc<Mutex<Stream<IofActor>>> {
        self.in_stream.clone()
    }

    pub fn share_ooa(&self) -> Arc<Mutex<Stream<OofActor>>> {
        self.out_stream.clone()
    }
}