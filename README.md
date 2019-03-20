# Human-Activity-Recognition
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{multicol}

\title{Human Activity Recognition\large \\ CSCI-B657: Computer Vision \\ Spring 2019}
\author{Darshan Shinde(dshinde)\\ Virendra Wali(vwali)\\  Bivas Maiti (bmaiti) }
\date{February 28,2019}
\usepackage{natbib}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{multicol}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=blue,      
    urlcolor=blue,
}
 \geometry{
 left=20mm,
 right=20mm,
 top=25mm,
 bottom=35mm
 }
\begin{document}

\maketitle
\section{Problem Statement}
Human Activity Recognition is the task of classifying different commonly carried out activities(like lying, standing, running, cycling, ascending stairs, etc.) based on sensory data from devices like accelerometer and gyroscope. It is an important task in the healthcare domain. By classifying activities of patients and monitoring their trends, valuable insights can be obtained. Most of today's smartphones come with IMUs (inertial measurement units), which have in-built accelerometer and gyroscope sensors. 
 We will be using the \href{https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring}{ PAMAP2 Physical Activity Monitoring Data Set} from the UCI Machine Learning Repository.This is the dataset which hold the entries for readings from different sensors. Using this dataset, we will try to answer a set of questions based on the data, and will also try to tackle the issue of user-dependency on the sensory data for different activities. 
\section{Data}
We are using PAMAP2 Dataset (Physical Activity Monitoring Data Set) from the UCI machine learning repository. This dataset is collected by setting 3 sensors at wrist, chest and ankle. 9 subjects (1 female and 8 male) are participated in the data collection. 18 different activities are performed by every subject and sensor readings are collected and mapped with activity ground truths.
Here are some points to mention about the dataset:
\begin{itemize}
    \item Data feature includes 9-axis IMU data streams for sensors on each of hand, chest, and ankle and subject heart rate.
    \item 1.9 million data points of 52 features each, spread over nine subjects.
    \item 18 different activity IDs, including sitting, walking, running, folding laundry, and cycling..9 million data points of 52 features each, spread over nine subjects.
    \item Total size of dataset: 1.61 GB
\end{itemize}

\section{Questions}
General perception is that there may exist a correlation between a subject and his/her activities. There can be two scenarios when we want to recognize the activity.
\begin{itemize}
    \item We have information about the subject.
    \item We don't have information about the subject.
\end{itemize}
We are looking towards human activity recognition as a multi-class classification problem. In the first scenario, we are trying to answer two different questions.
\begin{enumerate}
    \item For the given subject, what is the activity?
    \item For the given activity, who is the subject? 
    \\ \\
    For the second scenario, we will try to build a user-independent model for classification problem. We will try to generalize the model so that our model can be used for any user, whilst still achieving decent performance. So, our question is as follows:
    \item  Predict what is the activity without any prior information about the user.
\end{enumerate}
\section{Evaluation Criteria}
There are a lot of existing projects and papers which deal with HAR. We also have a publication for this very dataset, \href{https://dl.acm.org/citation.cfm?id=2413148}{``Creating and benchmarking a new dataset for physical activity monitoring" }, which can be used as a standard benchmark for the evaluation of our models.


\section{Timeline and Roles}
\begin{itemize}
    \item Week 1:
    \begin{itemize}
        \item Feature Extraction 
        \item Data Pre-Processing
    \end{itemize}
    \item Week 2-4:
    \begin{itemize}
        \item Fitting and building Different Models for all questions
    \end{itemize}
    \item Week 5:
    \begin{itemize}
        \item Comparison of different Models and evaluating performance
    \end{itemize}
    \item Week 6-7:
    \begin{itemize}
        \item Try to optimize the model that works best for the dataset for different questions, including user-independent one. 
        \item Generation of Reports
    \end{itemize}
    \item Week 8:
    \begin{itemize}
        \item If time permits, we want to check how we can achieve decent accuracy with minimal data, i.e. only heart rate monitor and 1 IMU.
    \end{itemize}
\end{itemize}

\begin{multicols}{3}

    \underline {bmaiti:} 
    \begin{enumerate}
        \item Data Preprocessing
        \item Build Model for Question 1
        \item Model Optimization
    \end{enumerate}
    \underline {dshinde:} 
    \begin{enumerate}
        \item Data Preprocessing
        \item Build Model for Question 2
        \item Model Optimization
    \end{enumerate}
     \underline {vwali:} 
    \begin{enumerate}
        \item Data Preprocessing
        \item Build Model for Question 3
        \item Model Optimization
    \end{enumerate}

\end{multicols}
\end{document}

