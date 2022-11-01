from pyueye import ueye
import numpy as np
from ctypes import *

class Camera:
    max_aoi_yx = np.array([0, 2048, 0, 2592], dtype=np.uint16)
    def __init__(self, identity):
        # REAL VALUES
        self.img = None
        self.identity = identity
        # Set in self.initialize() & self.set_aoi_and_initialize_img_mem()
        self.cam, self.nBitsPerPixel, self.pitch, self.bytes_per_pixel, self.m_nColorMode, self.image_ready_event, self.image_exposure_event = self.setup()

        #Set in below func
        self.pcImageMemory, self.camera_aoi, self.MemID = None, None, None
        self.set_aoi_and_initialize_img_mem()

    def set_aoi_and_initialize_img_mem(self, aoi_yx=max_aoi_yx):  # (x, y, width, height)
        # If memory is already initialized, destroys current memory
        if self.pcImageMemory is not None:
            ueye.is_FreeImageMem(self.cam, self.pcImageMemory, self.MemID)
        self.pcImageMemory, self.MemID = ueye.c_mem_p(),  ueye.int()

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

    def read_img(self):  # [X_additional_offset_, x_decrease_in_width, y_increase_in_offset, y_decrease_in_height]
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

def camera_child_process(end_exposure_mp, img_copied_mp, img_loaded_mp, shared_img_mp, shared_img_mp_shape, child_ready_mp, main_process_is_running):
    cam = Camera(1)
    img_view = np.ndarray(shared_img_mp_shape, buffer=shared_img_mp._obj, dtype=np.uint8)

    child_ready_mp.value = 1
    while main_process_is_running.value:
        if img_copied_mp.value == 1:
            img_copied_mp.value = 0
            cam.await_event(cam.image_exposure_event)
            end_exposure_mp.value = 1
            cam.await_event(cam.image_ready_event)
            img_view[0:] = cam.read_img()
            img_loaded_mp.value = 1