import cv2
import numpy as np


def classify(pth):
    feature = feature_extract(pth)
    p = svm.predict(feature)
    print p


def feature_extract(pth):
    im = cv2.imread(pth)
    return bowDiction.compute(to_gray(im), gen_kaze_features(to_gray(im))[0])


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (600, 450))
    return gray


def gen_kaze_features(gray_img):
    sift = cv2.KAZE_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc


BOW = cv2.BOWKMeansTrainer(37)

print "1"
prod1 = {}
for i in range(1, 51):  # 397
    if i % 7 == 0:
        continue
    img = cv2.imread('./Data/crocodile/image_' + ('%04d' % i) + '.jpg')
    gray = to_gray(img)
    kp, desc = gen_kaze_features(gray)
    prod1[i] = kp
    BOW.add(desc)

print "2"
prod2 = {}
for i in range(1, 51):  # 454
    if i % 7 == 0:
        continue
    img = cv2.imread('./Data/Leopards/image_' + ('%04d' % i) + '.jpg')
    gray = to_gray(img)
    kp, desc = gen_kaze_features(gray)
    prod2[i] = kp
    BOW.add(desc)

print "3"
sift2 = cv2.KAZE_create()
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(BOW.cluster())

train_desc = []
train_labels = []

print "4"
for key in prod1:
    cur_img = cv2.imread('./Data/crocodile/image_' + ('%04d' % key) + '.jpg')
    train_desc.extend(bowDiction.compute(to_gray(cur_img), prod1[key]))
    train_labels.append(1)

print "5"
for key in prod2:
    cur_img = cv2.imread('./Data/Leopards/image_' + ('%04d' % key) + '.jpg')
    train_desc.extend(bowDiction.compute(to_gray(cur_img), prod2[key]))
    train_labels.append(2)

print "6"
svm = cv2.ml.SVM_create()
svm.train(np.array(train_desc), cv2.ml.ROW_SAMPLE, np.array(train_labels))
svm.save('model.xml')

for key in range(1, 51):  # 397
    if key % 7 == 0:
        classify('./Data/Leopards/image_' + ('%04d' % key) + '.jpg')

print ''

for key in range(1, 51):  # 454
    if key % 7 == 0:
        classify('./Data/crocodile/image_' + ('%04d' % key) + '.jpg')

print ''


