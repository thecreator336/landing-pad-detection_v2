from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11s.pt")

    model.train(
        data        = "C:/yolo_dataset/dataset.yaml",
        epochs      = 100,
        imgsz       = 640,
        batch       = 8,
        name        = "landing_pad_v3",
        seed        = 42,
        save        = True,
        save_period = 10,
        workers     = 2,
    )

    print("\n✓ Training complete!")
    print("  Best weights: runs/detect/landing_pad_v3/weights/best.pt")