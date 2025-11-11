use crate::config::STREAM_CAPACITY;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::mem::MaybeUninit;

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
}