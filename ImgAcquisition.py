from pyueye import ueye
import numpy as np
from ctypes import *
from enum import IntEnum
import multiprocessing.sharedctypes as mpc
from multiprocessing import Process
import time
import cv2


class ChildCommIdxs(IntEnum):
    exposure_ended = 0
    img_copied = 1
    img_loaded = 2
    child_ready = 3
    main_process_running = 4

#For Testing
def show_img(img, resize=None):
    if resize:
        img_resize = cv2.resize(img, (int(img.shape[1] * resize), int(img.shape[0] * resize)))
        cv2.imshow('test_reads', img_resize)
    else:
        cv2.imshow('test_reads', img)

    cv2.moveWindow('test_reads', 0, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Camera:
    max_aoi_yx = np.array([0, 2048, 0, 2592], dtype=np.uint16)

    def __init__(self, identity):
        # REAL VALUES
        self.img = None
        self.identity = identity
        # Set in self.initialize() & self.set_aoi_and_initialize_img_mem()
        self.cam, self.nBitsPerPixel, self.pitch, self.bytes_per_pixel, self.m_nColorMode, self.image_ready_event, self.image_exposure_event = self.setup()

        # Set in below func
        self.pcImageMemory, self.camera_aoi, self.MemID = None, None, None
        self.set_aoi_and_initialize_img_mem()

    def set_aoi_and_initialize_img_mem(self, aoi_yx=max_aoi_yx):  # (x, y, width, height)
        # If memory is already initialized, destroys current memory
        if self.pcImageMemory is not None:
            ueye.is_FreeImageMem(self.cam, self.pcImageMemory, self.MemID)
        self.pcImageMemory, self.MemID = ueye.c_mem_p(), ueye.int()

        rectAOI = ueye.IS_RECT()
        rectAOI.s32Y.value, rectAOI.s32Height.value, rectAOI.s32X.value, rectAOI.s32Width.value = aoi_yx

        nRet = ueye.is_AOI(self.cam, ueye.IS_AOI_IMAGE_SET_AOI, rectAOI, ueye.sizeof(rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        nRet = ueye.is_AllocImageMem(self.cam, rectAOI.s32Width, rectAOI.s32Height, self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.cam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.cam, self.m_nColorMode)
                if nRet != ueye.IS_SUCCESS:
                    print("is_SetColorMode ERROR")

        nRet = ueye.is_InquireImageMem(self.cam, self.pcImageMemory, self.MemID, rectAOI.s32Width, rectAOI.s32Height, self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")
        else:
            print("Press q to leave the programm")
        self.camera_aoi = rectAOI

        # Starts/Restarts img capture as setting the aoi cancels current capture (I think..)
        nRet = ueye.is_CaptureVideo(self.cam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

    def get_img(self):
        return np.reshape(ueye.get_data(self.pcImageMemory, self.camera_aoi.s32Width, self.camera_aoi.s32Height, self.nBitsPerPixel, self.pitch, copy=False),
                          (self.camera_aoi.s32Height.value, self.camera_aoi.s32Width.value, self.bytes_per_pixel))

    def set_camera_id(self, cam_id):
        nRet = ueye.is_SetCameraID(self.cam, cam_id)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetCameraID ERROR")

    def exit(self):
        # Disables the cam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_FreeImageMem(self.cam, self.pcImageMemory, self.MemID)
        ueye.is_ExitCamera(self.cam)

    def setup(self):
        hCam = ueye.HIDS(self.identity)  # 0: first available camera;  1-254: The camera with the specified camera ID
        sInfo = ueye.SENSORINFO()
        cInfo = ueye.CAMINFO()
        pitch = ueye.INT()
        nBitsPerPixel = ueye.INT(24)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        m_nColorMode = ueye.INT()  # Y8/RGB16/RGB24/REG32
        # Starts the driver and establishes the connection to the camera

        nRet = ueye.is_InitCamera(hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")

        # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points_str to
        nRet = ueye.is_GetCameraInfo(hCam, cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")

        # You can query additional information about the sensor type used in the camera
        nRet = ueye.is_GetSensorInfo(hCam, sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

        # Set display mode to DIB
        nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetDisplayMode ERROR")

        # Set the right color mode
        if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            # setup_indicies the color depth to the current windows setting
            ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
            bytes_per_pixel = int(nBitsPerPixel / 8)
            print("IS_COLORMODE_BAYER: ", )
            print("\tm_nColorMode: \t\t", m_nColorMode)
            print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
            print()

        elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            m_nColorMode = ueye.IS_CM_BGRA8_PACKED
            nBitsPerPixel = ueye.INT(32)
            bytes_per_pixel = int(nBitsPerPixel / 8)
            print("IS_COLORMODE_CBYCRY: ", )
            print("\tm_nColorMode: \t\t", m_nColorMode)
            print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
            print()

        elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            m_nColorMode = ueye.IS_CM_MONO8
            nBitsPerPixel = ueye.INT(8)
            bytes_per_pixel = int(nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColorMode: \t\t", m_nColorMode)
            print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
            print()

        else:
            # for monochrome camera models use Y8 mode
            m_nColorMode = ueye.IS_CM_MONO8
            nBitsPerPixel = ueye.INT(8)
            bytes_per_pixel = int(nBitsPerPixel / 8)
            print("else")

        # Mirrors across x_axis for top_down View
        nRet = ueye.is_SetRopEffect(hCam, ueye.IS_SET_ROP_MIRROR_UPDOWN, c_int(1), c_int(0))
        if nRet != ueye.IS_SUCCESS:
            print("is_SetRopEffect error")

        # Maxs pixel clock
        nRet = ueye.is_PixelClock(hCam, ueye.IS_PIXELCLOCK_CMD_SET, ueye.int(400), sizeof(ueye.int()))
        if nRet != ueye.IS_SUCCESS:
            print("is_PixelClock ERROR")

        # Event 1 - End of exposure of image, used to sync load cell values with final
        # Event 2 - Image ready for access to memory, ensures no duplicate reads / blank reads
        for event in [ueye.IS_SET_EVENT_END_OF_EXPOSURE, ueye.IS_SET_EVENT_FRAME]:
            nRet = ueye.is_Event(hCam, ueye.IS_EVENT_CMD_INIT, ueye.IS_INIT_EVENT(nEvent=event), sizeof(ueye.IS_INIT_EVENT()))
            if nRet != ueye.IS_SUCCESS:
                print("IS_EVENT_CMD_INIT ERROR")

            nRet = ueye.is_Event(hCam, ueye.IS_EVENT_CMD_ENABLE, c_uint(event), sizeof(c_uint(event)))
            if nRet != ueye.IS_SUCCESS:
                print("IS_EVENT_CMD_ENABLE ERROR")

        image_exposure_event, image_ready_event = ueye.IS_WAIT_EVENT(nEvent=ueye.IS_SET_EVENT_END_OF_EXPOSURE, nTimeout=1000), \
                                                  ueye.IS_WAIT_EVENT(nEvent=ueye.IS_SET_EVENT_FRAME, nTimeout=1000)

        return hCam, nBitsPerPixel, pitch, bytes_per_pixel, m_nColorMode, image_ready_event, image_exposure_event

    def await_event(self, event):
        nret = None
        while nret != ueye.IS_SUCCESS:
            nret = ueye.is_Event(self.cam, ueye.IS_EVENT_CMD_WAIT, event, ueye.sizeof(event))


def camera_read_child(child_comm, img_mp, img_shape):
    cam = Camera(1)
    img_view = np.ndarray(img_shape, buffer=img_mp.get_obj(), dtype=np.uint8)
    child_comm[ChildCommIdxs.child_ready] = 1

    while child_comm[ChildCommIdxs.main_process_running]:
        if child_comm[ChildCommIdxs.img_copied] == 1:
            child_comm[ChildCommIdxs.img_copied] = 0
            cam.await_event(cam.image_exposure_event)
            child_comm[ChildCommIdxs.img_copied] = 1
            cam.await_event(cam.image_ready_event)
            img_view[0:] = cam.get_img()
            child_comm[ChildCommIdxs.img_ready.value] = 1


def get_hsv_from_bgr(bgr):
    return cv2.cvtColor(np.array([[bgr]]), cv2.COLOR_BGR2HSV).flatten()


def get_hsv_range(bgr: np.ndarray, hsv_range: (int, int, int)):
    color_hsv = get_hsv_from_bgr(bgr).astype(np.int16)
    hsv_ranges = np.vstack((color_hsv - hsv_range, color_hsv + hsv_range))
    max_hsv = np.array([179, 255, 255], dtype=np.uint8)

    # Sets min and Maxes
    hsv_ranges[0][np.argwhere(hsv_ranges[0] < 0)] = 0
    hsv_ranges[1][np.argwhere(hsv_ranges[1] > max_hsv).flatten()] = max_hsv[np.argwhere(hsv_ranges[1] > max_hsv).flatten()]

    return hsv_ranges.astype(np.uint8)

    pass


def get_camera_img(child_comm_mp, img_view, bgr_range):
    '''while not child_comm_mp[ChildCommIdxs.img_loaded]:
        pass
    img = img_view.copy()
    child_comm_mp[ChildCommIdxs.img_copied] = 1'''

    # For Testing
    img = np.zeros((Camera.max_aoi_yx[1], Camera.max_aoi_yx[3], 3), dtype=np.uint8)
    lin_count = 10
    y_lin, x_lin = np.linspace(0, Camera.max_aoi_yx[1], num=lin_count, dtype=np.uint16), np.linspace(0, Camera.max_aoi_yx[3], num=lin_count, dtype=np.uint16)

    for colors, lin_offset in zip(bgr_range, (4, 2)):
        x, y = int(x_lin[lin_count // lin_offset]), int(y_lin[lin_count // lin_offset])
        color = cv2.cvtColor(np.array([[np.mean(colors, axis=0).astype(np.uint8)]]), cv2.COLOR_HSV2BGR).flatten()
        cv2.circle(img, (x, y), 50, (int(color[0]), int(color[1]), int(color[2])), -1)
    return img





def np_dtype_from_maxval(max_val, signed=False):
    if not signed:
        if max_val <= 1:
            return np.bool_
        elif max_val <= 255:
            return np.uint8
        elif max_val <= 65535:
            return np.uint16
        elif max_val <= 4294967295:
            return np.uint32
        else:
            return np.uint64
    else:
        if max_val <= 127:
            return np.int8
        elif max_val <= 32767:
            return np.int16
        elif max_val <= 2147483647:
            return np.uint32
        else:
            return np.uint64


def return_contour_data_from_cam(child_comm_mp, img_view, hsv_ranges, desired_data_count, prompt, prompt_delay=1, all_valid=False, area_cutoffs=None):
    contour_areas = np.zeros((desired_data_count, hsv_ranges.shape[0]), dtype=np.uint16)
    counts = np.zeros(hsv_ranges.shape[0], dtype=np_dtype_from_maxval(desired_data_count))
    bboxs = np.zeros((hsv_ranges.shape[0], 4), dtype=np_dtype_from_maxval(np.max(Camera.max_aoi_yx.max()), signed=True))
    contours_all = [[] for _ in range(hsv_ranges.shape[0])]
    if area_cutoffs is None: area_cutoffs = np.zeros(hsv_ranges.shape[0], dtype=np.bool_)

    print(prompt), print(f'Starting data collection in {prompt_delay} seconds'), time.sleep(prompt_delay), print('Starting data collection')
    while np.any(counts != desired_data_count):
        idxs = np.argwhere(counts != desired_data_count).flatten()
        cam_img = cv2.cvtColor(get_camera_img(child_comm_mp, img_view, hsv_ranges), cv2.COLOR_BGR2HSV)
        contours, area = get_contours_from_hsv_img(cam_img, hsv_ranges[idxs], bboxs, area_cutoffs[idxs], all_valid)
        if contours:
            valid_contours_is = np.argwhere(area != 0).flatten()
            data_storage_idxs = idxs[valid_contours_is].flatten()
            for cnt_i, storage_i in zip(valid_contours_is, data_storage_idxs):
                contours_all[storage_i].append(contours[cnt_i])
            contour_areas[counts[data_storage_idxs], data_storage_idxs] = area[valid_contours_is]
            counts[data_storage_idxs] += 1

    contour_end_idxs = []
    for contour in contours_all:
        contour_end_idxs.append(np.cumsum([cnt.shape[0] for cnt in contour]))

    contour_end_idxs = np.column_stack(contour_end_idxs)
    contour_start_idxs = np.vstack((np.repeat(0, contour_end_idxs.shape[1]), contour_end_idxs[0:-1]))

    contour_idxs_all = np.dstack((np.expand_dims(contour_start_idxs, 1), np.expand_dims(contour_end_idxs, 1)))

    return [np.vstack(cnt_list) for cnt_list in contours], contour_idxs_all, contour_areas


def save_training_data(data_count):
    # Get the maximum size of each data array

    img_shape = (Camera.max_aoi_yx[1], Camera.max_aoi_yx[3], 3)
    child_comm_mp, img_mp = mpc.Array('B', len(ChildCommIdxs)), mpc.Array('B', int(np.product(img_shape)))
    img_view = np.ndarray(img_shape, dtype=np.uint8, buffer=img_mp._obj)
    camera_child = Process(target=camera_read_child, args=(child_comm_mp, img_mp, img_shape))
    # camera_child.start()

    foot_bgr_lr = np.array([[0, 0, 255],
                            [0, 128, 128]], dtype=np.uint8)
    foot_hsv_ranges_lr = np.stack((get_hsv_range(foot_bgr_lr[0], (10, 70, 70)),
                                   get_hsv_range(foot_bgr_lr[1], (10, 70, 70))))

    '''while child_comm_mp[ChildCommIdxs.child_ready] != 0:
        pass'''
    test_contour, test_idxs, test_area = return_contour_data_from_cam(child_comm_mp, img_view, foot_hsv_ranges_lr, 10, prompt='Test')
    print('b')


def get_cx_cy_from_contour(cnt) -> (int, int) or None:
    try:
        moments = cv2.moments(cnt)
        return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
    except ZeroDivisionError:
        return None


def get_contours_from_hsv_img(hsv_img, hsv_ranges, bboxs, area_cutoffs, valid_only_if_all_found) -> None or ((np.ndarray or None, [...]),
                                                                                                             (np.ndarray or None, [...])):
    contours_all, areas, = [], []

    for bbox, hsv_range in zip(bboxs, hsv_ranges):
        if bbox is not None: bbox = Camera.max_aoi_yx
        mask = cv2.inRange(hsv_img[bbox[0]:bbox[1], bbox[2]: bbox[3]], hsv_range[0], hsv_range[1])
        contours = list(cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0])

        if not contours:
            if valid_only_if_all_found:
                return False, False
            else:
                contours_all.append(None), areas.append(0)
                continue

        contours.sort(key=len)
        cx_cy = get_cx_cy_from_contour(contours[-1])

        if not cx_cy:
            if valid_only_if_all_found:
                return False, False
            else:
                contours_all.append(None), areas.append(0)
                continue

        contours_all.append(contours[-1] - cx_cy)
        areas.append(cv2.contourArea(contours[-1]))

    return contours_all, np.array(areas)


save_training_data(16)
