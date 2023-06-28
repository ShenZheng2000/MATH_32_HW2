# MATH_32_HW2

# NOTE
**For all questions related to this homework, please open a github issue so others with similar problems can benefit from it.**

Or you can contact the instructors via email: [Shen Zheng](mailto:shenzhen@andrew.cmu.edu), [Changjie Lu](mailto:lucha@kean.edu)

# Overview
In this assignment, you will implement a basic style transfer framework utilizing Gram matrices. The primary objectives are understanding how style and content can be manipulated within images and gaining experience in hyperparameter tuning. This task will enhance your comprehension of style transfer principles and aid in the development of your optimization skills.

# Installation
* Check [HW1](https://github.com/ShenZheng2000/MATH_32_HW1) for basic installations.
* Install Pillow using `pip install Pillow`
* Install [Git](https://git-scm.com/download/) and connect it to VS Code. Here is a [tutorial](https://stackoverflow.com/questions/42606837/how-do-i-use-bash-on-windows-from-the-visual-studio-code-integrated-terminal).

# Code Completion
* Open `style_and_content.py`, and complete 5 TODO's.
* The 1st and 2nd TODOs ask you to implement the content loss.
* The 3rd TODO asks you to implement the [gram matrix](https://en.wikipedia.org/wiki/Gram_matrix). 
* The 4th and 5th TODOs ask you to implement the style loss using the gram matrix.

# Hyperparameter Tuning
* Open `run.sh` and uncomment the lines under `############# Hyperparameter Tuning ##############`
* Run in terminal
  ```
  bash run.sh
  ```

# Exchange Content and Style
* Open `run.sh` and uncomment the lines under `############# Exchange Content and Style ##############`
* Run in terminal
  ```
  bash run.sh
  ```

# Style Transfer (Given Images)
* Open `run.sh` and uncomment the lines under `############# Style Transfer ##############`
* Run in terminal
  ```
  bash run.sh
  ```
  
# Style Transfer (Your Images)
* Prepare a content image and a style image on your own.
* Open `run.sh` and add a line like this:
  ```
  run_main $path_to_your_content_image $path_to_your_style_image
  ```
* Run the following command in the terminal:
  ```
  bash run.sh
  ```

# Results and Discussion
* For `Hyperparameter Tuning`
  * Which style weight (1e4, 1e5, 1e6) produces the best visual result?
  * Why isn't a larger style weight always preferable for style transfer?
* For `Exchange Content and Style`
  * How does the image appear when you exchange content and style?
  * Can you explain the types of images suitable for content/style?
* For `Style Transfer`
  * Does the style transfer result look visually pleasing?
  * If not, list at least three ways to improve it.

# Submission

* **NOTE: Updated 6/20/2023** 
* Navigate to the submission folder for HW2 at [here](https://drive.google.com/drive/folders/1SAUYLKyXfgZ7JiOueIIpis0e1t3axuah)
* Locate the folder with your name.
* Include all the results in a PDF file. 
* The PDF file should contain the following information, from top to bottom:
  * Your name, email, and student ID.
  * Codes for the five TODOs.
  * Images for ALL experiments (each row: content image, style image, and output image)
  * Short answers for ALL questions in the `Results and Discussion` section.  
* Submit the PDF file by placing it in your folder.

# Deadline
* ~~07/02/2023~~ **07/07/2023**
