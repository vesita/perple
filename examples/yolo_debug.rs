// 随机挑帧测 YOLO 输出
use perple::color::{load_model, image_to_tensor, image::scale_image, utils::to_input};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stems = ["000101","000205","000335","000408","000500",
                 "000569","000642","000153","000278","000601"];
    let mut session = load_model("model/quantized/yolo11n.onnx")?;

    for stem in &stems {
        let path = format!("data/labeled/camera/image/{}.jpg", stem);
        let img = image::open(&path)?;
        let (resized, msg) = scale_image(&img, 640, 640);
        let arr = image_to_tensor(&resized, 640, 640);
        let tensor = to_input(&arr);
        let outputs = session.run(ort::inputs!["images" => tensor])?;
        let extracted = outputs[0].try_extract_tensor::<f32>()?;
        let data = extracted.1;
        let stride = extracted.0[2] as usize;

        // 直接读 conf（假设 sigmoid 已内置）
        let mut dets: Vec<(f32, f32, f32, f32, f32)> = (0..stride)
            .map(|i| (data[4*stride+i], data[0*stride+i], data[1*stride+i], data[2*stride+i], data[3*stride+i]))
            .filter(|(c,_,_,_,_)| *c > 0.3).collect();
        dets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        println!("{}: conf>0.3={}, top3={:?}",
            stem, dets.len(),
            dets.iter().take(3).map(|(c,_,_,_,_)| format!("{:.4}",c)).collect::<Vec<_>>());
    }
    Ok(())
}
