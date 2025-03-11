import os

def list_committees_and_count(directory):
    committee_count = 0
    committees_path = os.path.join(directory, 'Committees')

    if os.path.exists(committees_path):
        for committee_name in os.listdir(committees_path):
            committee_path = os.path.join(committees_path, committee_name)
            if os.path.isdir(committee_path):
                print(f"Committee: {committee_name}")
                committee_count += 1

    print(f"Total number of committees: {committee_count}")


list_committees_and_count('data')