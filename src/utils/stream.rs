use crate::config::STREAM_CAPACITY;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::mem::MaybeUninit;
use std::boxed::Box;


/// 一个固定容量的线程安全流结构，用于在生产者和消费者之间传递数据
/// 推荐使用方法：
/// 获取写入位置的可变引用 -> 填充数据 -> 提交写入操作
/// 获取读取位置的引用 -> 处理数据 -> 提交读取操作


pub struct Stream<T: Default + Send + Clone> {
    pool: Vec<MaybeUninit<Option<T>>>,
    read_index: AtomicUsize,
    write_index: AtomicUsize,
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