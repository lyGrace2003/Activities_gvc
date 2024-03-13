import cv2
import numpy as np
import matplotlib.pyplot as plt

def activity_one():
    image_path = "Images/Image.png" 
    image = cv2.imread(image_path)

    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray_image)
    cv2.waitKey(0)

    row, col = 134, 210
    pixel_value = gray_image[row, col]
    print("Pixel value at ({}, {}):".format(row, col), pixel_value)

    _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Black and White Image", bw_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def activity_two():
    image = cv2.imread("Images/Image.png")

    height, width, _ = image.shape
    print("Image size: {} x {}".format(width, height))

    pixel = image[100, 100]
    print("Pixel value (R, G, B):", pixel)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(gray, 100, 200)
    edges3 = cv2.Canny(gray, 150, 250)
    edges4 = cv2.Canny(gray, 200, 300)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(edges1, cmap='gray')
    axs[0, 0].set_title('Edges 1')
    axs[0, 1].imshow(edges2, cmap='gray')
    axs[0, 1].set_title('Edges 2')
    axs[1, 0].imshow(edges3, cmap='gray')
    axs[1, 0].set_title('Edges 3')
    axs[1, 1].imshow(edges4, cmap='gray')
    axs[1, 1].set_title('Edges 4')

    plt.show()

    b, g, r = cv2.split(image)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(b, cmap='gray')
    axs[0].set_title('Blue Channel')
    axs[1].imshow(g, cmap='gray')
    axs[1].set_title('Green Channel')
    axs[2].imshow(r, cmap='gray')
    axs[2].set_title('Red Channel')

    plt.show()

def activity_three():
    image = cv2.imread("Images/Image.png")
    image2 = cv2.imread("Images/image2.jpg")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(image)

    hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist_gray2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    fig, axs = plt.subplots(2, 4, figsize=(12, 8))

    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original Image")

    axs[0, 1].imshow(gray_image, cmap='gray')
    axs[0, 1].set_title("Grayscale Image")

    axs[0, 2].plot(hist_gray)
    axs[0, 2].set_title("Histogram of Grayscale Image")
    axs[0, 2].set_xlabel("Pixel Value")
    axs[0, 2].set_ylabel("Frequency")

    axs[0, 3].plot(hist_r, color='red')
    axs[0, 3].plot(hist_g, color='green')
    axs[0, 3].plot(hist_b, color='blue')
    axs[0, 3].set_title("RGB Histogram")
    axs[0, 3].set_xlabel("Pixel Value")
    axs[0, 3].set_ylabel("Frequency")
    axs[0, 3].legend(['Red', 'Green', 'Blue'])

    axs[1, 0].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Original Image 2")

    axs[1, 1].imshow(gray_image2, cmap='gray')
    axs[1, 1].set_title("Grayscale Image 2")

    axs[1, 2].plot(hist_gray2)
    axs[1, 2].set_title("Histogram of Grayscale Image 2")
    axs[1, 2].set_xlabel("Pixel Value")
    axs[1, 2].set_ylabel("Frequency")

    axs[1, 3].axis('off')

    plt.tight_layout()
    plt.show()

def activity_four():
    image_path = "Images/image2.jpg"

    image = cv2.imread(image_path)
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(232)
    plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
    plt.title("GrayScale Image")

    plt.subplot(233)
    plt.hist(gray_image.ravel(), 256, [0, 256])
    plt.title("Grayscale Histogram")

    edges = cv2.Canny(gray_image, 100, 200)
    plt.subplot(234)
    plt.imshow(edges, cmap='gray')
    plt.title("Edges")

    plt.tight_layout()
    plt.show()

def activity_five():
    original_image = cv2.imread('Images/Image2.jpg')
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(original_image, 100, 200)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')

    axs[0, 1].imshow(edges, cmap='gray')
    axs[0, 1].set_title('Edges')

    axs[1, 0].imshow(gray_image, cmap='gray')
    axs[1, 0].set_title('Grayscale Image')

    axs[1, 1].plot(histogram)
    axs[1, 1].set_title('Histogram')

    print('Filename:', 'path_to_original_image.jpg')
    print('Format:', 'RGB')
    print('Width:', original_image.shape[1])
    print('Height:', original_image.shape[0])
    print('Size:', original_image.size)

    pixel_value = original_image[1060, 1240]
    print('Pixel value at (100, 100):', pixel_value)

    plt.show()

def display_menu():
    print("Computer Vision Menu:")
    print("1. Activity One")
    print("2. Activity Two")
    print("3. Activity Three")
    print("4. Activity Four")
    print("5. Activity Five")
    print("6. Exit")

def main():
    while True:
        display_menu()
        choice = input("Enter your choice: ")
        
        if choice == '1':
            activity_one()
        elif choice == '2':
            activity_two()
        elif choice == '3':
            activity_three()
        elif choice == '4':
            activity_four()
        elif choice == '5':
            activity_five()
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()
