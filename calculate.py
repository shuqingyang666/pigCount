from ultralytics import YOLO

def main():
    model = YOLO(r"D:\pig\ultralytics-8.3.225\ultralytics-8.3.225\runs\detect\train5\weights\best.pt")

    results = model.val(
        data=r"D:\pig\pig_dataset6\pig_dataset6.yaml",
        imgsz=736,
        workers=0      # ✅ Windows 避免多进程出错
    )

    print("\n====== Final Evaluation Metrics ======")
    print("Precision:", results.box.p.mean())
    print("Recall:", results.box.r.mean())
    print("mAP50:", results.box.map50)
    print("mAP50-95:", results.box.map)
    print("Per-class mAP:", results.box.maps)
    print("Speed:", results.speed)

if __name__ == "__main__":   # ✅ 必须加在这里！
    main()
