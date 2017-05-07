# -*- coding: utf-8 -*-
import cv2
import numpy as np


class WHUCaptcha(object):


    @staticmethod
    def join_rect(rect1, rect2):
        if rect1[0] > rect2[0]:
            x = rect2[0]
            w = max(rect2[2], rect1[0] - rect2[0] + rect1[2])
        else:
            x = rect1[0]
            w = max(rect1[2], rect2[0] - rect1[0] + rect2[2])
        if rect1[1] > rect2[1]:
            y = rect2[1]
            h = max(rect2[3], rect1[1] - rect2[1] + rect1[3])
        else:
            y = rect1[1]
            h = max(rect1[3], rect2[1] - rect1[1] + rect2[3])
        return x, y, w, h

    @staticmethod
    def color_filter(bgr_img):
        # bgr筛选出r值较低的噪点
        # 因为存在r值低但是饱和度高的噪点会混入后续hsv筛出的字符中去
        upper_not_red_bgr = np.array([255, 255, 120])
        lower_not_red_bgr = np.array([0, 0, 0])
        bgr_filter_res = cv2.inRange(bgr_img, lower_not_red_bgr, upper_not_red_bgr)
        # hsv筛出字符的大致轮廓
        # 因为字符的饱和度显著高于其他色块
        # opencv不知为何hsv的取值有些奇怪不是[360, 100, 100]而是[180, 255, 255]
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        upper_s_hsv = np.array([180, 255, 255])
        lower_s_hsv = np.array([0, 150, 0])
        hsv_filter_res = cv2.inRange(hsv_img, lower_s_hsv, upper_s_hsv)
        # 将两次的筛选合并输出最后的二值化字符图像
        filter_res = cv2.subtract(hsv_filter_res, bgr_filter_res)
        return filter_res

    @staticmethod
    def split_characters(binary_img):
        """
        分离出字符的像素块
        :param binary_img:
        :return:
        """
        # opencv画出轮廓,转化为方框并判定面积过小的像素块为噪点并消去，并标记不能为一个独立字符的像素块等待后续合并
        filter_res, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        individual_flags = []
        for contour in contours:
            rect = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 0.5:
                cv2.rectangle(filter_res, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 0, -1)
            else:
                rectangles.append(rect)
                if cv2.contourArea(contour) < 8:
                    individual_flags.append(False)
                else:
                    individual_flags.append(True)
        # 将被标记不能为一个独立字符的像素块找到最近的像素块进行合并
        merged_rectangles = []
        used_rect_index = []
        for flag_index, flag in enumerate(individual_flags):
            if not flag:
                rect_to_merge = rectangles[flag_index]
                test_x_middle_point = rect_to_merge[0] + rect_to_merge[2] / 2
                # 这个只是list里面只有一个元素的时候报merge_index未定义的错误
                merge_index = flag_index
                min_distance = 100
                for rect_index, rect in enumerate(rectangles):
                    if flag_index != rect_index:
                        candidate_x_middle_point = rect[0] + rect[2] / 2
                        distance = abs(test_x_middle_point - candidate_x_middle_point)
                        if distance < min_distance:
                            min_distance = distance
                            merge_index = rect_index
                merged_rectangles.append(WHUCaptcha.join_rect(rect_to_merge, rectangles[merge_index]))
                used_rect_index.append(flag_index)
                used_rect_index.append(merge_index)
        for rect_index, rect in enumerate(rectangles):
            if rect_index not in used_rect_index:
                merged_rectangles.append(rect)
        characters = []
        merged_rectangles.sort()
        for rect in merged_rectangles:
            characters.append(binary_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])
        return characters

    @staticmethod
    def normalize_characters(characters):
        normalized_characters = []
        for character in characters:
            normalized_characters.append(cv2.resize(character, (28, 28), interpolation=cv2.INTER_NEAREST))
        return normalized_characters

    @staticmethod
    def pre_process(img_dir='captcha.png'):
        """
        预处理加切分字符串
        :param img_dir:
        :return: 28*28二值化的字符块列表
        """
        # 注意opencv是用bgr而不是rgb打开图片的
        bgr_img = cv2.imread(img_dir)
        filter_res = WHUCaptcha.color_filter(bgr_img)
        characters = WHUCaptcha.split_characters(filter_res)
        normalized_characters = WHUCaptcha.normalize_characters(characters)
        return normalized_characters

    @staticmethod
    def pre_process_demo_version(img_dir='captcha.png'):
        """
        内部过程可展示版本的pre_process()
        实际运行效果可以参见example/pre_process*.png
        :param img_dir:
        :return:
        """
        import copy
        # 注意opencv是用bgr而不是rgb打开图片的
        bgr_img = cv2.imread(img_dir)
        upper_not_red_bgr = np.array([255, 255, 120])
        lower_not_red_bgr = np.array([0, 0, 0])
        bgr_filter_res = cv2.inRange(bgr_img, lower_not_red_bgr, upper_not_red_bgr)
        # hsv筛出字符的大致轮廓
        # 因为字符的饱和度显著高于其他色块
        # opencv不知为何hsv的取值有些奇怪不是[360, 100, 100]而是[180, 255, 255]
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        upper_s_hsv = np.array([180, 255, 255])
        lower_s_hsv = np.array([0, 150, 0])
        hsv_filter_res = cv2.inRange(hsv_img, lower_s_hsv, upper_s_hsv)
        # 将两次的筛选合并输出最后的二值化字符图像
        filter_res = cv2.subtract(hsv_filter_res, bgr_filter_res)
        # opencv画出轮廓,转化为方框并，判定面积过小的像素块为噪点并消去
        filter_res, contours, hierarchy = cv2.findContours(filter_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        clear_noise_demo = copy.deepcopy(filter_res)
        after_clear_noise = copy.deepcopy(filter_res)
        individual_flags = []
        for contour in contours:
            rect = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 0.5:
                cv2.rectangle(clear_noise_demo, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 100, 0)
                cv2.rectangle(after_clear_noise, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 0, -1)
            else:
                rectangles.append(rect)
                if cv2.contourArea(contour) < 8:
                    individual_flags.append(False)
                else:
                    individual_flags.append(True)
        before_merge_demo = copy.deepcopy(after_clear_noise)
        after_merge_demo = copy.deepcopy(after_clear_noise)
        for rect in rectangles:
            cv2.rectangle(before_merge_demo, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 100, 0)
        merged_rectangles = []
        used_rect_index = []
        for flag_index, flag in enumerate(individual_flags):
            if not flag:
                rect_to_merge = rectangles[flag_index]
                test_x_middle_point = rect_to_merge[0] + rect_to_merge[2] / 2
                # 这个只是list里面只有一个元素的时候报merge_index未定义的错误
                merge_index = flag_index
                min_distance = 100
                for rect_index, rect in enumerate(rectangles):
                    if flag_index != rect_index:
                        candidate_x_middle_point = rect[0] + rect[2] / 2
                        distance = abs(test_x_middle_point-candidate_x_middle_point)
                        if distance < min_distance:
                            min_distance = distance
                            merge_index = rect_index
                merged_rectangles.append(WHUCaptcha.join_rect(rect_to_merge, rectangles[merge_index]))
                used_rect_index.append(flag_index)
                used_rect_index.append(merge_index)
        for rect_index, rect in enumerate(rectangles):
            if rect_index not in used_rect_index:
                merged_rectangles.append(rect)
        rectangles = merged_rectangles
        for rect in rectangles:
            cv2.rectangle(after_merge_demo, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 100, 0)
        from matplotlib import pyplot as plt
        (b, g, r) = cv2.split(bgr_img)
        rgb_img = cv2.merge([r, g, b])
        plt.subplot(811)
        plt.imshow(rgb_img)
        plt.subplot(812)
        plt.imshow(hsv_img)
        plt.subplot(813)
        plt.imshow(bgr_filter_res, 'gray_r')
        plt.subplot(814)
        plt.imshow(hsv_filter_res, 'gray_r')
        plt.subplot(815)
        plt.imshow(filter_res, 'gray_r')
        plt.subplot(816)
        plt.imshow(clear_noise_demo, 'gray_r')
        plt.subplot(817)
        plt.imshow(before_merge_demo, 'gray_r')
        plt.subplot(818)
        plt.imshow(after_merge_demo, 'gray_r')
        mng = plt.get_current_fig_manager()
        mng.resize(*(500, 1000))
        plt.show()
