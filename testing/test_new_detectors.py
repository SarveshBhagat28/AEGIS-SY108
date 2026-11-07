
import pandas as pd
from pprint import pprint

from detection.file_activity_detector import evaluate_file_activity
from detection.device_activity_detector import evaluate_device_activity


EMPLOYEE = "TEST001"
DATE = "2010-08-18"


def create_data(file_covered=True, device_covered=True,
                removable=0, deletes=0, connects=1, disconnects=0):

    return pd.DataFrame([{
        "user": EMPLOYEE,
        "calendar_date": pd.Timestamp(DATE),
        "file_day_covered": file_covered,
        "removable_media_events": removable,
        "file_deletes": deletes,
        "device_day_covered": device_covered,
        "device_connects": connects,
        "device_disconnects": disconnects
    }])


def run_test(name, function, data):
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    result = function(data, EMPLOYEE, DATE)
    pprint(result)


def main():

    data = create_data()
    run_test("1. Normal file activity", evaluate_file_activity, data)

    data = create_data(removable=3)
    run_test("2. High removable media activity", evaluate_file_activity, data)

    data = create_data(deletes=25)
    run_test("3. High file deletion activity", evaluate_file_activity, data)

    data = create_data(connects=1)
    run_test("4. Normal device activity", evaluate_device_activity, data)

    data = create_data(connects=6)
    run_test("5. High device connection activity", evaluate_device_activity, data)

    data = create_data(removable=1)
    run_test("6. Moderate removable media activity", evaluate_file_activity, data)

    data = create_data(deletes=10)
    run_test("7. Moderate file deletion activity", evaluate_file_activity, data)

    data = create_data(connects=2)
    run_test("8. Moderate device connection activity", evaluate_device_activity, data)

    data = create_data(file_covered=False)
    run_test("9. Missing file telemetry coverage", evaluate_file_activity, data)

    data = create_data(device_covered=False)
    run_test("10. Missing device telemetry coverage", evaluate_device_activity, data)

    data = create_data()
    data = data[data["user"] != EMPLOYEE]
    run_test("11. Missing employee-day record", evaluate_file_activity, data)


if __name__ == "__main__":
    main()
