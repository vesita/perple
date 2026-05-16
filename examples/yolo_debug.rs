use perple::color::{
    load_model, image_to_tensor,
    image::scale_image,
    utils::to_input,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = image::open("data/cloud/camera/000101.jpg")?;
    let (resized, _msg) = scale_image(&img, 640, 640);
    let arr = image_to_tensor(&resized, 640, 640);
    let tensor = to_input(&arr);
    let mut session = load_model("model/quantized/yolo11n.onnx")?;
    let outputs = session.run(ort::inputs!["images" => tensor])?;
    
    let extracted = outputs[0].try_extract_tensor::<f32>()?;
    let shape: Vec<i64> = extracted.0.to_vec();
    let data = extracted.1;
    let stride = shape[2] as usize;  // 8400
    let ch = shape[1] as usize;      // 84
    
    println!("shape: {:?}", shape);
    
    // Collect detections with multi-class decoding
    let mut dets: Vec<(usize, u32, f32)> = Vec::new();
    for i in 0..stride {
        let mut best_cls = 0u32;
        let mut best_logit = data[4*stride + i];
        for c in 1..(ch-4) {
            let logit = data[(4+c)*stride + i];
            if logit > best_logit {
                best_logit = logit;
                best_cls = c as u32;
            }
        }
        let conf = 1.0 / (1.0 + (-best_logit).exp());
        if conf > 0.5 {
            dets.push((i, best_cls, conf));
        }
    }
    dets.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    
    println!("Detections > 0.5: {} (clipped to 16 by capacity)", dets.len());
    for (idx, cls, conf) in dets.iter().take(16) {
        let cx = data[0*stride + idx];
        let cy = data[1*stride + idx];
        let w  = data[2*stride + idx];
        let h  = data[3*stride + idx];
        let x1 = (cx - w/2.0) as u32;
        let y1 = (cy - h/2.0) as u32;
        let x2 = (cx + w/2.0) as u32;
        let y2 = (cy + h/2.0) as u32;
        println!("  cls={:>3} conf={:.3} xyxy=[{} {} {} {}] (cxcywh=[{:.0} {:.0} {:.0} {:.0}])",
            cls, conf, x1, y1, x2, y2, cx, cy, w, h);
    }
    
    Ok(())
}
