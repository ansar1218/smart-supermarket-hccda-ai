"""Huawei Cloud OCR General Table (Lab 3), as a reusable helper.

Needs three environment variables (never hard-code keys):
  HUAWEICLOUD_SDK_AK, HUAWEICLOUD_SDK_SK, HUAWEICLOUD_PROJECT_ID  (AP-Singapore)
and a General Table subscription in the OCR console, region AP-Singapore.
"""
import base64
import os

import pandas as pd

KEYS = ["HUAWEICLOUD_SDK_AK", "HUAWEICLOUD_SDK_SK", "HUAWEICLOUD_PROJECT_ID"]


def missing_keys():
    return [k for k in KEYS if not os.environ.get(k)]


def recognize_general_table(image_bytes):
    """Call RecognizeGeneralTable with a Base64 image; returns the response as a dict."""
    from huaweicloudsdkcore.auth.credentials import BasicCredentials
    from huaweicloudsdkcore.http.http_config import HttpConfig
    from huaweicloudsdkocr.v1 import GeneralTableRequestBody, OcrClient, RecognizeGeneralTableRequest
    from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion

    credentials = BasicCredentials(os.environ["HUAWEICLOUD_SDK_AK"], os.environ["HUAWEICLOUD_SDK_SK"],
                                   os.environ["HUAWEICLOUD_PROJECT_ID"])
    client = (OcrClient.new_builder()
              .with_http_config(HttpConfig.get_default_config())
              .with_credentials(credentials)
              .with_region(OcrRegion.value_of("ap-southeast-3"))
              .build())
    request = RecognizeGeneralTableRequest()
    request.body = GeneralTableRequestBody(image=base64.b64encode(image_bytes).decode(), return_excel=True)
    return client.recognize_general_table(request).to_dict()


def to_dataframe(result):
    """Turn the table regions of an OCR result into a DataFrame (first row = header)."""
    cells = {}
    for region in result["result"]["words_region_list"]:
        if region["type"] != "table":
            continue
        for block in region["words_block_list"]:
            cells[(block["rows"][0], block["columns"][0])] = block["words"]
    if not cells:
        return pd.DataFrame()
    n_r = max(r for r, _ in cells) + 1
    n_c = max(c for _, c in cells) + 1
    grid = [[cells.get((r, c), "") for c in range(n_c)] for r in range(n_r)]
    return pd.DataFrame(grid[1:], columns=grid[0])


def excel_bytes(result):
    """The Base64 Excel file the API returns when return_excel=True, decoded."""
    b64 = result["result"].get("excel")
    return base64.b64decode(b64) if b64 else None
