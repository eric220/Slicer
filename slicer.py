from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Tuple, Union

import numpy as np

class InferenceSlicer:
    def __init__(
        self,
        callback: Callable[[np.ndarray], 'Detections'],
        slice_wh: Tuple[int, int] = (320, 320),
        overlap_ratio_wh: Tuple[float, float] = (0.2, 0.2),
        overlap_filter_strategy: Union[str, str] = None, #OverlapFilter.NON_MAX_SUPPRESSION,
        grid: Tuple[int, int] = None,
        iou_threshold: float = 0.5,
        thread_workers: int = 1,
    ):
        #overlap_filter_strategy = validate_overlap_filter(overlap_filter_strategy)
        self.slice_wh = slice_wh
        self.overlap_ratio_wh = overlap_ratio_wh
        self.iou_threshold = iou_threshold
        self.overlap_filter_strategy = overlap_filter_strategy
        self.callback = callback
        self.grid = grid
        self.thread_workers = thread_workers

    def __call__(self, image: np.ndarray):# -> Detections:
        """
        Performs slicing-based inference on the provided image using the specified
            callback.

        Args:
            image (np.ndarray): The input image on which inference needs to be
                performed. The image should be in the format
                `(height, width, channels)`.

        Returns:
            Detections: A collection of detections for the entire image after merging
                results from all slices and applying NMS.

        Example:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread(SOURCE_IMAGE_PATH)
            model = YOLO(...)

            def callback(image_slice: np.ndarray) -> sv.Detections:
                result = model(image_slice)[0]
                return sv.Detections.from_ultralytics(result)

            slicer = sv.InferenceSlicer(
                callback=callback,
                overlap_filter_strategy=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            )

            detections = slicer(image)
            ```
        """
        detections_list = []
        resolution_wh = (image.shape[1], image.shape[0])
        if self.grid:
            offsets = self._generate_grid_offsets(
                resolution_wh=resolution_wh,
                overlap_ratio_wh=self.overlap_ratio_wh,
                grid=self.grid,
            )
        else:
            offsets = self._generate_offset(
                resolution_wh=resolution_wh,
                slice_wh=self.slice_wh,
                overlap_ratio_wh=self.overlap_ratio_wh,
            )
        return offsets
        """
        with ThreadPoolExecutor(max_workers=self.thread_workers) as executor:
            futures = [
                executor.submit(self._run_callback, image, offset) for offset in offsets
            ]
            for future in as_completed(futures):
                detections_list.append(future.result())

        merged = Detections.merge(detections_list=detections_list)
        if self.overlap_filter_strategy == OverlapFilter.NONE:
            return merged
        elif self.overlap_filter_strategy == OverlapFilter.NON_MAX_SUPPRESSION:
            return merged.with_nms(threshold=self.iou_threshold)
        elif self.overlap_filter_strategy == OverlapFilter.NON_MAX_MERGE:
            return merged.with_nmm(threshold=self.iou_threshold)
        else:
            warnings.warn(
                f"Invalid overlap filter strategy: {self.overlap_filter_strategy}",
                category=SupervisionWarnings,
            )
            return merged
            """

    def _run_callback(self, image, offset):# -> Detections:
        """
        Run the provided callback on a slice of an image.

        Args:
            image (np.ndarray): The input image on which inference needs to run
            offset (np.ndarray): An array of shape `(4,)` containing coordinates
                for the slice.

        Returns:
            Detections: A collection of detections for the slice.
        """
        image_slice = crop_image(image=image, xyxy=offset)
        detections = self.callback(image_slice)
        resolution_wh = (image.shape[1], image.shape[0])
        detections = move_detections(
            detections=detections, offset=offset[:2], resolution_wh=resolution_wh
        )

        return detections

    @staticmethod
    def _generate_offset(
        resolution_wh: Tuple[int, int],
        slice_wh: Tuple[int, int],
        overlap_ratio_wh: Tuple[float, float],
    ) -> np.ndarray:
        """
        Generate offset coordinates for slicing an image based on the given resolution,
        slice dimensions, and overlap ratios.

        Args:
            resolution_wh (Tuple[int, int]): A tuple representing the width and height
                of the image to be sliced.
            slice_wh (Tuple[int, int]): A tuple representing the desired width and
                height of each slice.
            overlap_ratio_wh (Tuple[float, float]): A tuple representing the desired
                overlap ratio for width and height between consecutive slices. Each
                value should be in the range [0, 1), where 0 means no overlap and a
                value close to 1 means high overlap.

        Returns:
            np.ndarray: An array of shape `(n, 4)` containing coordinates for each
                slice in the format `[xmin, ymin, xmax, ymax]`.

        Note:
            The function ensures that slices do not exceed the boundaries of the
                original image. As a result, the final slices in the row and column
                dimensions might be smaller than the specified slice dimensions if the
                image's width or height is not a multiple of the slice's width or
                height minus the overlap.
        """
        slice_width, slice_height = slice_wh
        image_width, image_height = resolution_wh
        overlap_ratio_width, overlap_ratio_height = overlap_ratio_wh

        width_stride = slice_width - int(overlap_ratio_width * slice_width)
        height_stride = slice_height - int(overlap_ratio_height * slice_height)

        ws = np.arange(0, image_width, width_stride)
        hs = np.arange(0, image_height, height_stride)

        xmin, ymin = np.meshgrid(ws, hs)
        xmax = np.clip(xmin + slice_width, 0, image_width)
        ymax = np.clip(ymin + slice_height, 0, image_height)

        offsets = np.stack([xmin, ymin, xmax, ymax], axis=-1).reshape(-1, 4)

        return offsets

    @staticmethod
    def _generate_grid_offsets(
        resolution_wh: Tuple[int, int],
        overlap_ratio_wh: Tuple[float, float],
        grid: Tuple[int, int]
    ) -> np.ndarray:
        """
        Generate offset coordinates for slicing an image based on the requested grid shape
        and overlap ratios.

        Args:
            resolution_wh (Tuple[int, int]): A tuple representing the width and height
                of the image to be sliced.
            overlap_ratio_wh (Tuple[float, float]): A tuple representing the desired
                overlap ratio for width and height between consecutive slices. Each
                value should be in the range [0, 1), where 0 means no overlap and a
                value close to 1 means high overlap.
            grid (Tuple[int, int]): A tuple representing the desired number of tiles.

        Returns:
            np.ndarray: An array of shape `(n, 4)` containing coordinates for each
                slice in the format `[xmin, ymin, xmax, ymax]`.
        """
        image_width, image_height = resolution_wh
        overlap_ratio_width, overlap_ratio_height = overlap_ratio_wh
        y_split, x_split = grid
        #image_width = 2*stride-overlap+(x_split-2)(stride-overlap)
        stride_width = np.ceil((image_width/(2-overlap_ratio_width+(x_split-2)*(1-overlap_ratio_width))))
        stride_height = np.ceil((image_height/(2-overlap_ratio_height+(y_split-2)*(1-overlap_ratio_height))))
        overlap_width = stride_width*overlap_ratio_width
        overlap_height = stride_height*overlap_ratio_height
        ws = np.arange(0, image_width, stride_width-overlap_width, dtype=int)[:x_split]
        hs = np.arange(0, image_height, stride_height-overlap_height, dtype=int)[:y_split]
        xmin, ymin = np.meshgrid(ws, hs)
        xmax = np.clip(xmin + stride_width, 0, image_width)
        ymax = np.clip(ymin + stride_height, 0, image_height)

        return np.stack([xmin, ymin, xmax, ymax], axis=-1).reshape(-1, 4)