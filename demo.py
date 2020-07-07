from PIL import Image
from makeup import Inference

def main(
    source_path='./makeup/xfsy_0106.png',
    reference_path='./makeup/vFG586.png',
    save_path='./makeup/transferred_image.png',
    speed=False,
    device="cpu"):

    # Using cpu for inference
    inference = Inference(device)

    source = Image.open(source_path).convert("RGB")
    reference = Image.open(reference_path)

    # Transfer the makeup from reference to source.
    image, face = inference.transfer(source, reference, with_face=True)
    image.save(save_path)
   
    if speed:
        import time
        start = time.time()
        for _ in range(100):
            inference.transfer(source, reference)
        print("Time cost for 100 iters: ", time.time() - start)


if __name__ == '__main__':
    main()
