import cv2
import numpy as np
import glob
import os


def calibrate_stereo_cameras(
    left_image_folder,
    right_image_folder,
    aruco_dict_type,
    board_squares_x,
    board_squares_y,
    square_length,
    marker_length,
):
    """
    Calibrates a stereo camera pair using a ChArUco board.

    Args:
        left_image_folder (str): Path to the folder containing left camera images.
        right_image_folder (str): Path to the folder containing right camera images.
        aruco_dict_type (str): The ArUco dictionary type (e.g., 'DICT_4X4_50').
        board_squares_x (int): Number of squares in the X direction of the ChArUco board.
        board_squares_y (int): Number of squares in the Y direction of the ChArUco board.
        square_length (float): Length of the ChArUco square in meters.
        marker_length (float): Length of the ArUco marker in meters.

    Returns:
        tuple: A tuple containing the stereo calibration results:
               - mtxL (np.array): Left camera matrix.
               - distL (np.array): Left camera distortion coefficients.
               - mtxR (np.array): Right camera matrix.
               - distR (np.array): Right camera distortion coefficients.
               - R (np.array): Rotation matrix between the two cameras.
               - T (np.array): Translation vector between the two cameras.
               - E (np.array): Essential matrix.
               - F (np.array): Fundamental matrix.
    """

    # 1. Define the ArUco dictionary and ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_type))
    # board = cv2.aruco.CharucoBoard_create(
    #     board_squares_x, board_squares_y, square_length, marker_length, aruco_dict
    # )

    board = cv2.aruco.CharucoBoard(
        (board_squares_x, board_squares_y), square_length, marker_length, aruco_dict
    )

    # 2. Get image paths and prepare lists for calibration data
    print(f"Loading left data from {os.path.join(left_image_folder)}")
    left_images = sorted(glob.glob(os.path.join(left_image_folder, "*.jpg")))
    right_images = sorted(glob.glob(os.path.join(right_image_folder, "*.jpg")))

    print(f"Loaded {len(left_images)} images")

    all_charuco_corners_L = []
    all_charuco_ids_L = []
    all_charuco_corners_R = []
    all_charuco_ids_R = []

    image_size = None

    # 3. Detect ChArUco corners in all images
    print("Detecting ChArUco corners...")
    for i in range(len(left_images)):
        if i > 100:
            break
        img_L = cv2.imread(left_images[i])
        img_R = cv2.imread(right_images[i])

        if image_size is None:
            image_size = img_L.shape[:2]

        corners_L, ids_L, _ = cv2.aruco.detectMarkers(img_L, aruco_dict)
        corners_R, ids_R, _ = cv2.aruco.detectMarkers(img_R, aruco_dict)

        if ids_L is not None and ids_R is not None:
            ret_L, charuco_corners_L, charuco_ids_L = (
                cv2.aruco.interpolateCornersCharuco(corners_L, ids_L, img_L, board)
            )
            ret_R, charuco_corners_R, charuco_ids_R = (
                cv2.aruco.interpolateCornersCharuco(corners_R, ids_R, img_R, board)
            )

            if ret_L > 8 and ret_R > 8:
                all_charuco_corners_L.append(charuco_corners_L)
                all_charuco_ids_L.append(charuco_ids_L)
                all_charuco_corners_R.append(charuco_corners_R)
                all_charuco_ids_R.append(charuco_ids_R)

    # 4. Calibrate each camera individually
    print("Calibrating left camera...")
    print(len(all_charuco_corners_L))
    print(len(all_charuco_ids_L))
    ret_L, mtxL, distL, rvecsL, tvecsL = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners_L, all_charuco_ids_L, board, image_size, None, None
    )

    print("Calibrating right camera...")
    print(len(all_charuco_corners_R), len(all_charuco_ids_R))
    ret_R, mtxR, distR, rvecsR, tvecsR = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners_R, all_charuco_ids_R, board, image_size, None, None
    )

    # 5. Calibrate the stereo pair
    print("Calibrating stereo pair...")

    print(len(all_charuco_corners_L))
    print(len(all_charuco_ids_R))
    charuco_obj_points = board.getChessboardCorners()
    # We must provide object points and image points for the stereoCalibrate function.
    # The ChArUco board object points are the same for both cameras.

    left_corners_sampled = []
    right_corners_sampled = []
    obj_pts = []

    for i in range(len(all_charuco_ids_L)):
        left_sub_corners = []
        right_sub_corners = []
        obj_pts_sub = []

        for j in range(len(all_charuco_ids_L[i])):
            idx = np.where(all_charuco_ids_R[i] == all_charuco_ids_L[i][j])
            if idx[0].size == 0:
                continue
            left_sub_corners.append(all_charuco_corners_L[i][j])
            right_sub_corners.append(all_charuco_corners_R[i][idx])
            obj_pts_sub.append(charuco_obj_points[all_charuco_ids_L[i][j]])
        if len(left_sub_corners) > 3 and len(right_sub_corners) > 3:
            obj_pts.append(np.array(obj_pts_sub, dtype=np.float32))
            left_corners_sampled.append(np.array(left_sub_corners, dtype=np.float32))
            right_corners_sampled.append(np.array(right_sub_corners, dtype=np.float32))

            # corners_left_decimate.append()
        else:
            print("FAILURE")
            continue

    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        obj_pts,
        left_corners_sampled,
        right_corners_sampled,
        mtxL,
        distL,
        mtxR,
        distR,
        image_size,
        # criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        # flags=cv2.CALIB_FIX_INTRINSIC,
    )

    print("Stereo Calibration Complete!")
    return mtxL, distL, mtxR, distR, R, T, E, F


if __name__ == "__main__":
    ###### CHANGE THESE VALUES TO POINT TO YOUR DIRECTORY
    LEFT_FOLDER = "/Users/khalilholiday/Downloads/Calibration_left/"
    RIGHT_FOLDER = "/Users/khalilholiday/Downloads/Calibration_right/"
    ######

    ##### THESE SHOULD NOT CHANGE #########
    ARUCO_DICT_TYPE = "DICT_5X5_1000"
    BOARD_SQUARES_X = 12
    BOARD_SQUARES_Y = 9
    SQUARE_LENGTH_M = 0.03  # in meters
    MARKER_LENGTH_M = 0.022  # in meters
    #####################################

    mtxL, distL, mtxR, distR, R, T, E, F = calibrate_stereo_cameras(
        LEFT_FOLDER,
        RIGHT_FOLDER,
        ARUCO_DICT_TYPE,
        BOARD_SQUARES_X,
        BOARD_SQUARES_Y,
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
    )

    print("\n--- Calibration Results ---")
    print("Left Camera Matrix (mtxL):\n", mtxL)
    print("Left Distortion Coeffs (distL):\n", distL)
    print("Right Camera Matrix (mtxR):\n", mtxR)
    print("Right Distortion Coeffs (distR):\n", distR)
    print("Rotation Matrix (R):\n", R)
    print("Translation Vector (T):\n", T)
    print("Essential Matrix (E):\n", E)
    print("Fundamental Matrix (F):\n", F)

 # Requested function 1: Draw the charuco points for the left and right camera. Output to local folder

def draw_charuco_points(
    left_image_folder,
    right_image_folder,
    aruco_dict_type,
    board_squares_x,
    board_squares_y,
    square_length,
    marker_length,
    output_dir="charuco_outputs"
):
    """
    Draw the detected ArUco markers and ChArUco corners for each left/right image
    and save annotated images to a local folder (without changing your calibration code).

    Args:
        left_image_folder (str): Path to left images.
        right_image_folder (str): Path to right images.
        aruco_dict_type (str): e.g., 'DICT_5X5_1000'.
        board_squares_x (int): ChArUco board squares in X.
        board_squares_y (int): ChArUco board squares in Y.
        square_length (float): Square size in meters.
        marker_length (float): Marker size in meters.
        output_dir (str): Folder to save outputs (created if missing).

    Returns:
        dict: Summary with counts and output paths.
    """
     
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_type))
    board = cv2.aruco.CharucoBoard(
        (board_squares_x, board_squares_y), square_length, marker_length, aruco_dict
    )

    
    os.makedirs(output_dir, exist_ok=True)
    left_out = os.path.join(output_dir, "left")
    right_out = os.path.join(output_dir, "right")
    os.makedirs(left_out, exist_ok=True)
    os.makedirs(right_out, exist_ok=True)

     
    def list_images(folder):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder, e)))
        return sorted(files)

    left_images = list_images(left_image_folder)
    right_images = list_images(right_image_folder)

    if len(left_images) == 0 or len(right_images) == 0:
        print("No images found in one or both folders.")
        return {"left_saved": 0, "right_saved": 0, "output_dir": output_dir}

    n = min(len(left_images), len(right_images))
    saved_left = 0
    saved_right = 0

    def annotate_and_save(img_path, save_dir, side_tag):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}")
            return False

        
        corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict)
        vis = img.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

            
            ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, img, board
            )
            
            if ch_corners is not None and ch_ids is not None and len(ch_ids) > 0:
                try:
                    cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
                except TypeError:
                    
                    cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners)

        fname = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(save_dir, f"{fname}_{side_tag}_charuco.jpg")
        cv2.imwrite(out_path, vis)
        return True

    for i in range(n):
        if annotate_and_save(left_images[i], left_out, "left"):
            saved_left += 1
        if annotate_and_save(right_images[i], right_out, "right"):
            saved_right += 1

    print(f"Saved {saved_left} left and {saved_right} right annotated images to '{output_dir}'.")
    return {"left_saved": saved_left, "right_saved": saved_right, "output_dir": output_dir}

    

    def _charuco_board_from_params(aruco_dict_type, squares_x, squares_y, square_len_m, marker_len_m):
        aruco_dict = getattr(cv2.aruco, aruco_dict_type)
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y,
            square_len_m, marker_len_m,
            dictionary
        )
        return dictionary, board

    def _detect_charuco_points(img_bgr, dictionary, board):
        """Return (charuco_corners Nx2 float32, charuco_ids Nx1 int) or (None, None) if not enough points."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        if ids is None or len(ids) == 0:
            return None, None

        try:
            
            cv2.aruco.refineDetectedMarkers(
                image=gray, board=board,
                detectedCorners=corners, detectedIds=ids,
                rejectedCorners=None
            )
        except Exception:
            pass

        
        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=gray, board=board
        )
        if ret is None or ch_corners is None or ch_ids is None or len(ch_ids) == 0:
            return None, None

        
        ch_corners = ch_corners.reshape(-1, 2).astype(np.float32)
        return ch_corners, ch_ids.astype(np.int32)

    def _match_charuco_by_id(ptsL, idsL, ptsR, idsR):
        """
        Given per-image charuco corners + ids for left and right, align by shared IDs.
        Returns (matched_ptsL Nx2, matched_ptsR Nx2). If no overlap, returns (None, None).
        """
        if ptsL is None or ptsR is None or idsL is None or idsR is None:
            return None, None

        
        mapL = {int(i[0]): idx for idx, i in enumerate(idsL)}
        mapR = {int(i[0]): idx for idx, i in enumerate(idsR)}
        shared = sorted(set(mapL.keys()) & set(mapR.keys()))
        if len(shared) == 0:
            return None, None

        mL = np.array([ptsL[mapL[k]] for k in shared], dtype=np.float32)
        mR = np.array([ptsR[mapR[k]] for k in shared], dtype=np.float32)
        return mL, mR

    def _list_images(folder):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder, f"*{e}")))
        files.sort()
        return files
    
    # Build board/dictionary once for detection
    dictionary, board = _charuco_board_from_params(
        ARUCO_DICT_TYPE, BOARD_SQUARES_X, BOARD_SQUARES_Y,
        SQUARE_LENGTH_M, MARKER_LENGTH_M
    )

    # List and align the left/right image files by sorted order 
    left_image_paths  = _list_images(LEFT_FOLDER)
    right_image_paths = _list_images(RIGHT_FOLDER)
    num_pairs = min(len(left_image_paths), len(right_image_paths))


    OUT_DIR = os.path.join(LEFT_FOLDER, "..", "matched_viz")
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)



    USE_EPILINES = True if F is not None else False

    saved = 0
    for i in range(num_pairs):
        L_path = left_image_paths[i]
        R_path = right_image_paths[i]

        L_img = cv2.imread(L_path, cv2.IMREAD_COLOR)
        R_img = cv2.imread(R_path, cv2.IMREAD_COLOR)
        if L_img is None or R_img is None:
            continue

        
        ptsL, idsL = _detect_charuco_points(L_img, dictionary, board)
        ptsR, idsR = _detect_charuco_points(R_img, dictionary, board)

        
        mL, mR = _match_charuco_by_id(ptsL, idsL, ptsR, idsR)
        if mL is None or mR is None or len(mL) < 6:
            continue

         
        out_name = os.path.splitext(os.path.basename(L_path))[0] + "_matched.png"
        out_path = draw_matched_stereo_points(
            left_img_path=L_path,
            right_img_path=R_path,
            pts_left=mL,
            pts_right=mR,
            out_dir=OUT_DIR,
            filename=out_name,
            max_points=300,               
            draw_indices=False,
            draw_connections=True,
            show_epilines=USE_EPILINES,
            F=F if USE_EPILINES else None,
            point_radius=4,
            thickness=2,
            gap_px=8,
        )
        print("Saved:", out_path)
        saved += 1

    if saved == 0:
        print("No matched stereo visualizations were saved (not enough overlapping Charuco IDs found).")
    else:
        print(f"Saved {saved} matched stereo visualizations to: {OUT_DIR}")


    # Requested function 3: Find the position of the left camera using "estimatePoseCharucoBoard" (https://docs.opencv.org/4.12.0/d9/d6a/group__aruco.html#ga21b51b9e8c6422a4bac27e48fa0a150b)

    # Requested function 4: Draw the position of left and right camera and save that data locally or display from function
