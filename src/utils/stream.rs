use crate::config::STREAM_CAPACITY;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::mem::MaybeUninit;


/// 一个固定容量的线程安全流结构，用于在生产者和消费者之间传递数据
/// 推荐使用方法：
/// 获取写入位置的可变引用 -> 填充数据 -> 提交写入操作
/// 获取读取位置的引用 -> 处理数据 -> 提交读取操作


pub struct Stream<T: Default + Send> {
    pool: [MaybeUninit<Option<T>>; STREAM_CAPACITY],
    read_index: AtomicUsize,
    write_index: AtomicUsize,
}

impl<T: Default + Send> Stream<T> { 
    pub fn new() -> Self {
        // 创建一个未初始化的数组
        let mut pool: [MaybeUninit<Option<T>>; STREAM_CAPACITY] = 
            unsafe { MaybeUninit::uninit().assume_init() };
        
        // 初始化所有元素
        for i in 0..STREAM_CAPACITY {
            pool[i] = MaybeUninit::new(None);
        }
        
        Self {
            pool,
            read_index: AtomicUsize::new(0),
            write_index: AtomicUsize::new(0),
        }
    }
    
    /// 获取写入位置的可变引用，如果缓冲区满了则返回Err
    pub fn get_write_mut(&mut self) -> Result<&mut Option<T>, &'static str> {
        let current_read = self.read_index.load(Ordering::Acquire);
        let current_write = self.write_index.load(Ordering::Acquire);
        
        let next_index = (current_write + 1) % STREAM_CAPACITY;
        if next_index == current_read {
            return Err("缓冲区已满");
        }
        
        // 安全地返回可变引用
        unsafe {
            Ok(&mut *self.pool[current_write].as_mut_ptr())
        }
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
            return None; // 队列为空
        }
        
        // 安全地返回引用
        unsafe {
            Some(&*self.pool[current_read].as_ptr())
        }
    }
    
    /// 提交读取操作，将读索引向前移动
    pub fn commit_read(&mut self) -> Result<(), &'static str> {
        let current_read = self.read_index.load(Ordering::Acquire);
        let current_write = self.write_index.load(Ordering::Acquire);
        
        if current_read == current_write {
            return Err("缓冲区为空");
        }
        
        let next_index = (current_read + 1) % STREAM_CAPACITY;
        // 更新读索引
        self.read_index.store(next_index, Ordering::Release);
        Ok(())
    }
    
    pub fn write(&mut self, item: T) -> Result<(), &'static str> {
        loop {
            let current_read = self.read_index.load(Ordering::Acquire);
            let current_write = self.write_index.load(Ordering::Acquire);
            
            let next_index = (current_write + 1) % STREAM_CAPACITY;
            if next_index == current_read {
                return Err("缓冲区已满");
            }
            
            // 尝试更新写索引
            if self.write_index.compare_exchange(
                current_write, 
                next_index, 
                Ordering::Release, 
                Ordering::Relaxed
            ).is_ok() {
                // 安全地写入数据
                unsafe {
                    self.pool[current_write].as_mut_ptr().write(Some(item));
                }
                return Ok(());
            }
            // 如果更新失败，重新尝试
        }
    }
    
    pub fn read(&mut self) -> Option<T> {
        loop {
            let current_read = self.read_index.load(Ordering::Acquire);
            let current_write = self.write_index.load(Ordering::Acquire);
            
            if current_read == current_write {
                return None; // 队列为空
            }
            
            let next_index = (current_read + 1) % STREAM_CAPACITY;
            
            // 尝试更新读索引
            if self.read_index.compare_exchange(
                current_read, 
                next_index, 
                Ordering::Release, 
                Ordering::Relaxed
            ).is_ok() {
                // 安全地读取数据
                let item = unsafe {
                    self.pool[current_read].assume_init_read()
                };
                return item;
            }
            // 如果更新失败，重新尝试
        }
    }
    
    /// 检查流中是否有数据
    pub fn has_data(&self) -> bool {
        let current_read = self.read_index.load(Ordering::Acquire);
        let current_write = self.write_index.load(Ordering::Acquire);
        current_read != current_write
    }
    
    /// 直接写入到指定索引位置，无额外拷贝
    /// 通过读写标记保障数据一致性
    pub fn write_direct<F>(&mut self, writer: F) -> Result<(), &'static str>
    where
        F: FnOnce(&mut Option<T>),
    {
        loop {
            let current_read = self.read_index.load(Ordering::Acquire);
            let current_write = self.write_index.load(Ordering::Acquire);
            
            let next_index = (current_write + 1) % STREAM_CAPACITY;
            if next_index == current_read {
                return Err("缓冲区已满");
            }
            
            // 尝试更新写索引
            if self.write_index.compare_exchange(
                current_write, 
                next_index, 
                Ordering::Release, 
                Ordering::Relaxed
            ).is_ok() {
                // 直接操作数据
                unsafe {
                    writer(&mut *self.pool[current_write].as_mut_ptr());
                }
                return Ok(());
            }
        }
    }
}

impl<T: Default + Send + Clone> Stream<T> {
    // 克隆实现等其他方法...
}