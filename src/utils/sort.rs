

pub fn quick_sort(arr: &mut [i32]) {
if arr.len() <= 1 {
    return;
}
let pivot = arr[arr.len() / 2];
let mut left = 0;
let mut right = arr.len() - 1;
while left <= right {
    while arr[left] < pivot {
        left += 1;
    }
    while arr[right] > pivot {
        right -= 1;
    }
    if left <= right {
        arr.swap(left, right);
        left += 1;
        right -= 1;
    }
}
quick_sort(&mut arr[..right + 1]);
quick_sort(&mut arr[left..]);
}

pub fn group_sort<T: Ord>(arr: &mut [T], split: usize, offset: usize) {
    if arr.len() < split || offset >= split {
        return;
    }
    let m = arr.len() % split;
    if m != 0 {
        return;
    }
    let length = arr.len() / split;
    let pivot_index = (length - 1) * split + offset;
    let mut left = 0;
    let mut right = length - 1;
    while left <= right {
        while arr[left * split + offset] < arr[pivot_index] {
            left += 1;
        }
        while arr[right * split + offset] > arr[pivot_index] {
            right -= 1;
        }
        if left <= right {
            for order in 0..split {
                arr.swap(left * split + order, right * split + order);
            }
            left += 1;
            right -= 1;
        }
    }
    group_sort(&mut arr[..right * split + offset], split, offset);
    group_sort(&mut arr[left * split + offset..], split, offset);
}

pub fn group_sort_by<T, F>(arr: &mut [T], split: usize, offset: usize, compare: F)
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    if arr.len() < split || offset >= split {
        return;
    }
    let m = arr.len() % split;
    if m != 0 {
        return;
    }

    // 使用Vec作为栈来模拟递归
    let mut stack = vec![(0, arr.len() / split)];
    
    while let Some((start, end)) = stack.pop() {
        if start >= end {
            continue;
        }
        
        let length = end - start;
        if length <= 1 {
            continue;
        }
        
        let pivot_index = (start + length - 1) * split + offset;
        let mut left = start;
        let mut right = end - 1;
        
        while left <= right {
            // 向右移动左指针，直到找到一个不小于pivot的元素
            while left <= right {
                match compare(&arr[left * split + offset], &arr[pivot_index]) {
                    std::cmp::Ordering::Less => left += 1,
                    _ => break,
                }
            }
            
            // 向左移动右指针，直到找到一个不大于pivot的元素
            while left <= right {
                match compare(&arr[right * split + offset], &arr[pivot_index]) {
                    std::cmp::Ordering::Greater => {
                        if right == 0 { 
                            break; 
                        }
                        right -= 1;
                    }
                    _ => break,
                }
            }
            
            if left <= right {
                for order in 0..split {
                    arr.swap(left * split + order, right * split + order);
                }
                left += 1;
                if right > 0 {
                    right -= 1;
                } else {
                    break;
                }
            }
        }
        
        // 避免无效的递归调用
        if right > start {
            stack.push((start, right + 1));
        }
        if left < end {
            stack.push((left, end));
        }
    }
}