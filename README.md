# wgisd-Faster-RCNN
MSc Project


#Does not work could just call one image?
def visual_test(model, data_loader_test):
    model.eval()
    device = "cpu:0"
    model.to(device)
    images, targets = next(iter(data_loader_test))
    images = list(image.to(device) for image in images)
    outputs = model(images)
    detection_threshold = 0.5
    print(images)
    sample = images[0].permute(1,2,0).cpu().numpy()
    boxes = outputs[0]['boxes'].data.cpu().numpy()
    scores = outputs[0]['scores'].data.cpu().numpy()
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    print(boxes)
    for box in boxes:
        cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 2)
    ax.set_axis_off()
    ax.imshow(sample)
