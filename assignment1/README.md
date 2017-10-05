Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2017.

CS231n Convolutional Neural Networks for Visual Recognition
In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
understand the train/val/test splits and the use of validation data for hyperparameter tuning.
develop proficiency in writing efficient vectorized code with numpy
implement and apply a k-Nearest Neighbor (kNN) classifier
implement and apply a Multiclass Support Vector Machine (SVM) classifier
implement and apply a Softmax classifier
implement and apply a Two layer neural network classifier
understand the differences and tradeoffs between these classifiers
get a basic understanding of performance improvements from using higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)
Setup

You can work on the assignment in one of two ways: locally on your own machine, or on a virtual machine on Google Cloud.

Working remotely on Google Cloud (Recommended)

Note: after following these instructions, make sure you go to Download data below (you can skip the Working locally section).

As part of this course, you can use Google Cloud for your assignments. We recommend this route for anyone who is having trouble with installation set-up, or if you would like to use better CPU/GPU resources than you may have locally. Please see the set-up tutorial here for more details. :)

Working locally

Get the code as a zip file here. As for the dependencies:

Installing Python 3.5+: To use python3, make sure to install version 3.5 or 3.6 on your local machine. If you are on Mac OS X, you can do this using Homebrew with brew install python3. You can find instructions for Ubuntu here.

Virtual environment: If you decide to work locally, we recommend using virtual environment for the project. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run the following:

cd assignment1
sudo pip install virtualenv      # This may already be installed
virtualenv -p python3 .env       # Create a virtual environment (python3)
# Note: you can also use "virtualenv .env" to use your default python (usually python 2.7)
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
# Work on the assignment for a while ...
deactivate                       # Exit the virtual environment
Note that every time you want to work on the assignment, you should run source .env/bin/activate (from within your assignment1 folder) to re-activate the virtual environment, and deactivate again whenever you are done.

Download data:

Once you have the starter code (regardless of which method you choose above), you will need to download the CIFAR-10 dataset. Run the following from the assignment1 directory:

cd cs231n/datasets
./get_datasets.sh
Start IPython:

After you have the CIFAR-10 data, you should start the IPython notebook server from the assignment1 directory, with the jupyter notebook command. (See the Google Cloud Tutorial for any additional steps you may need to do for setting this up, if you are working remotely)

If you are unfamiliar with IPython, you can also refer to our IPython tutorial.

Some Notes

NOTE 1: This year, the assignment1 code has been tested to be compatible with python versions 2.7, 3.5, 3.6 (it may work with other versions of 3.x, but we won’t be officially supporting them). You will need to make sure that during your virtualenv setup that the correct version of python is used. You can confirm your python version by (1) activating your virtualenv and (2) running which python.

NOTE 2: If you are working in a virtual environment on OSX, you may potentially encounter errors with matplotlib due to the issues described here. In our testing, it seems that this issue is no longer present with the most recent version of matplotlib, but if you do end up running into this issue you may have to use the start_ipython_osx.sh script from the assignment1 directory (instead of jupyter notebook above) to launch your IPython notebook server. Note that you may have to modify some variables within the script to match your version of python/installation directory. The script assumes that your virtual environment is named .env.

Submitting your work:

Whether you work on the assignment locally or using Google Cloud, once you are done working run the collectSubmission.sh script; this will produce a file called assignment1.zip. Please submit this file on Canvas.

Q1: k-Nearest Neighbor classifier (20 points)

The IPython Notebook knn.ipynb will walk you through implementing the kNN classifier.

Q2: Training a Support Vector Machine (25 points)

The IPython Notebook svm.ipynb will walk you through implementing the SVM classifier.

Q3: Implement a Softmax classifier (20 points)

The IPython Notebook softmax.ipynb will walk you through implementing the Softmax classifier.

Q4: Two-Layer Neural Network (25 points)

The IPython Notebook two_layer_net.ipynb will walk you through the implementation of a two-layer neural network classifier.

Q5: Higher Level Representations: Image Features (10 points)

The IPython Notebook features.ipynb will walk you through this exercise, in which you will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.

Q6: Cool Bonus: Do something extra! (+10 points)

Implement, investigate or analyze something extra surrounding the topics in this assignment, and using the code you developed. For example, is there some other interesting question we could have asked? Is there any insightful visualization you can plot? Or anything fun to look at? Or maybe you can experiment with a spin on the loss function? If you try out something cool we’ll give you up to 10 extra points and may feature your results in the lecture.

 cs231n
 cs231n
karpathy@cs.stanford.edu


http://cs231n.github.io/gce-tutorial/

CS231n Convolutional Neural Networks for Visual Recognition
Google Cloud Tutorial
Google Cloud Tutorial

BEFORE WE BEGIN

BIG REMINDER: Make sure you stop your instances!

(We know you won’t read until the very bottom once your assignment is running, so we are printing this at the top too since it is super important)

Don’t forget to stop your instance when you are done (by clicking on the stop button at the top of the page showing your instances), otherwise you will run out of credits and that will be very sad. :(

If you follow our instructions below correctly, you should be able to restart your instance and the downloaded software will still be available.


Create and Configure Your Account

For the class project and assignments, we offer an option to use Google Compute Engine for developing and testing your implementations. This tutorial lists the necessary steps of working on the assignments using Google Cloud. We expect this tutorial to take about an hour. Don’t get intimidated by the steps, we tried to make the tutorial detailed so that you are less likely to get stuck on a particular step. Please tag all questions related to Google Cloud with google_cloud on Piazza.

This tutorial goes through how to set up your own Google Compute Engine (GCE) instance to work on the assignments. Each student will have $100 in credit throughout the quarter. When you sign up for the first time, you also receive $300 credits from Google by default. Please try to use the resources judiciously. But if $100 ends up not being enough, we will try to adjust this number as the quarter goes on. Note: for assignment 1, we are only supporting python version 2.7 (the default installation from the script) and 3.5.3.

First, if you don’t have a Google Cloud account already, create one by going to the Google Cloud homepage and clicking on Compute. When you get to the next page, click on the blue TRY IT FREE button. If you are not logged into gmail, you will see a page that looks like the one below. Sign into your gmail account or create a new one if you do not already have an account.


If you already have a gmail account, it will direct you to a signup page which looks like the following.


Click the appropriate yes or no button for the first option, and check yes for the latter two options after you have read the required agreements. Press the blue Agree and continue button to continue to the next page to enter the requested information (your name, billing address and credit card information). Once you have entered the required information, press the blue Start my free trial button. You will be greeted by a page like this:


To change the name of your project, click on Manage project settings on the Project info button and save your changes.


Launch a Virtual Instance

To launch a virtual instance, go to the Compute Engine menu on the left column of your dashboard and click on VM instances. Then click on the blue CREATE button on the next page. This will take you to a page that looks like the screenshot below. (NOTE: Please carefully read the instructions in addition to looking at the screenshots. The instructions tell you exactly what values to fill in).


Make sure that the Zone is set to be us-west1-b (especially for assignments where you need to use GPU instances). Under Machine type pick the 8 vCPUs option. Click on the customize button under Machine type and make sure that the number of cores is set to 8 and the number of GPUs is set to None (we will not be using GPUs in assignment 1 and this tutorial will be updated with instructions for GPU usage). Click on the Change button under Boot disk, choose OS images, check Ubuntu 16.04 LTS and click on the blue select button. Check Allow HTTP traffic and Allow HTTPS traffic. Click on disk and then Disks and uncheck Delete boot disk when instance is deleted (Note that the “Disks” option may be hiding under an expandable URL at the bottom of that webform). Click on the blue Create button at the bottom of the page. You should have now successfully created a Google Compute Instance, it might take a few minutes to start running. Your screen should look something like the one below. When you want to stop running the instance, click on the blue stop button above.


Take note of your <YOUR-INSTANCE-NAME>, in this case, my instance name is instance-2.

Connect to Your Virtual Instance and Download the Assignment

Now that you have created your virtual GCE, you want to be able to connect to it from your computer. The rest of this tutorial goes over how to do that using the command line. First, download the Google Cloud SDK that is appropriate for your platform from here and follow their installation instructions. NOTE: this tutorial assumes that you have performed step #4 on the website which they list as optional. When prompted, make sure you select us-west1-b as the time zone. The easiest way to connect is using the gcloud compute command below. The tool takes care of authentication for you. On OS X, run:

./<DIRECTORY-WHERE-GOOGLE-CLOUD-IS-INSTALLED>/bin/gcloud compute ssh --zone=us-west1-b <YOUR-INSTANCE-NAME>
See this page for more detailed instructions. You are now ready to work on the assignments on Google Cloud.

Run the following command to download the current assignment onto your GCE:

wget http://cs231n.stanford.edu/assignments/2017/spring1617_assignment1.zip 
Then run:

sudo apt-get install unzip
and

unzip spring1617_assignment1.zip
to get the contents. You should now see a folder titled assignmentX. To install the necessary dependencies for assignment 1 (NOTE: you only need to do this for assignment 1), cd into the assignment directory and run the provided shell script: (Note: you will need to hit the [enter] key at all the “[Y/n]” prompts)

cd assignment1 
./setup_googlecloud.sh
You will be prompted to enter Y/N at various times during the download. Press enter for every prompt. You should now have all the software you need for assignmentX. If you had no errors, you can proceed to work with your virtualenv as normal.

I.e. run

source .env/bin/activate
in your assignment directory to load the venv, and run

deactivate
to exit the venv. See assignment handout for details.

NOTE: The instructions above will run everything needed using Python 2.7. If you would like to use Python 3.5 instead, edit setup_googlecloud.sh to replce the line

virtualenv .env 
with

virtualenv -p python3 .env
before running

./setup_googlecloud.sh
Using Jupyter Notebook with Google Compute Engine

Many of the assignments will involve using Jupyter Notebook. Below, we discuss how to run Jupyter Notebook from your GCE instance and use it on your local browser.

Getting a Static IP Address

Change the Extenal IP address of your GCE instance to be static (see screenshot below).


To Do this, click on the 3 line icon next to the Google Cloud Platform button on the top left corner of your screen, go to Networking and External IP addresses (see screenshot below).


To have a static IP address, change Type from Ephemeral to Static. Enter your preffered name for your static IP, mine is assignment-1 (see screenshot below). And click on Reserve. Remember to release the static IP address when you are done because according to this page Google charges a small fee for unused static IPs. Type should now be set to Static.


Take note of your Static IP address (circled on the screenshot below). I used 104.196.224.11 for this tutorial.


Adding a Firewall rule

One last thing you have to do is adding a new firewall rule allowing TCP acess to a particular <PORT-NUMBER>. I usually use 7000 or 8000 for <PORT-NUMBER>. Click on the 3 line icon at the top of the page next to Google Cloud Platform. On the menu that pops up on the left column, go to Networking and Firewall rules (see the screenshot below).


Click on the blue CREATE FIREWALL RULE button. Enter whatever name you want: I used assignment1-rules. Enter 0.0.0.0/0 for Source IP ranges and tcp:<PORT-NUMBER> for Allowed protocols and ports where <PORT-NUMBER> is the number you used above. Click on the blue Create button. See the screen shot below.


NOTE: Some people are seeing a different screen where instead of Allowed protocols and ports there is a field titled Specified protocols and ports. You should enter tcp:<PORT-NUMBER> for this field if this is the page you see. Also, if you see a field titled Targets select All instances in the network.

Configuring Jupyter Notebook

The following instructions are excerpts from this page that has more detailed instructions.

On your GCE instance check where the Jupyter configuration file is located:

ls ~/.jupyter/jupyter_notebook_config.py
Mine was in /home/timnitgebru/.jupyter/jupyter_notebook_config.py

If it doesn’t exist, create one:

# Remember to activate your virtualenv ('source .env/bin/activate') so you can actually run jupyter :)
jupyter notebook --generate-config
Using your favorite editor (vim, emacs etc…) add the following lines to the config file, (e.g.: /home/timnitgebru/.jupyter/jupyter_notebook_config.py):

c = get_config()

c.NotebookApp.ip = '*'

c.NotebookApp.open_browser = False

c.NotebookApp.port = <PORT-NUMBER>
Where <PORT-NUMBER> is the same number you used in the prior section. Save your changes and close the file.

Launching and connecting to Jupyter Notebook

The instructions below assume that you have SSH’d into your GCE instance using the prior instructions, have already downloaded and unzipped the current assignment folder into assignmentX (where X is the assignment number), and have successfully configured Jupyter Notebook.

If you are not already in the assignment directory, cd into it by running the following command:

cd assignment1 
If you haven’t already done so, activate your virtualenv by running:

source .env/bin/activate
Launch Jupyter notebook using:

jupyter-notebook --no-browser --port=<PORT-NUMBER> 
Where <PORT-NUMBER> is what you wrote in the prior section.

On your local browser, if you go to http://<YOUR-EXTERNAL-IP-ADDRESS>:<PORT-NUMBER>, you should see something like the screen below. My value for <YOUR-EXTERNAL-IP-ADDRESS> was 104.196.224.11 as mentioned above. You should now be able to start working on your assignments.


Submission: Transferring Files From Your Instance To Your Computer

Once you are done with your assignments, run the submission script in your assignment folder. For assignment1, this will create a zip file called assignment1.zip containing the files you need to upload to Canvas. If you’re not in the assignment1 directory already, CD into it by running

cd assignment1
install zip by running

sudo apt-get install zip
and then run

bash collectSubmission.sh 
to create the zip file that you need to upload to canvas. Then copy the file to your local computer using the gcloud compute copy-file command as shown below. NOTE: run this command on your local computer:

gcloud compute copy-files [INSTANCE_NAME]:[REMOTE_FILE_PATH] [LOCAL_FILE_PATH]
For example, to copy my files to my desktop I ran:

gcloud compute copy-files instance-2:~/assignment1/assignment1.zip ~/Desktop
Another (perhaps easier) option proposed by a student is to directly download the zip file from Jupyter. After running the submission script and creating assignment1.zip, you can download that file directly from Jupyter. To do this, go to Jupyter Notebook and click on the zip file (in this case assignment1.zip). The file will be downloaded to your local computer.

Finally, remember to upload the zip file containing your submission to Canvas. (You can unzip the file locally if you want to double check your ipython notebooks and other code files are correctly inside).

You can refer to this page for more details on transferring files to/from Google Cloud.

BIG REMINDER: Make sure you stop your instances!

Don’t forget to stop your instance when you are done (by clicking on the stop button at the top of the page showing your instances). You can restart your instance and the downloaded software will still be available.

 cs231n
 cs231n
karpathy@cs.stanford.edu
