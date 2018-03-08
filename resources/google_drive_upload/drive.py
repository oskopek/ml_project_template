import httplib2
import os
from dotmap import DotMap

from apiclient import discovery, errors
from apiclient.http import MediaFileUpload
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/ml_template_gdrive.json
SCOPES = "https://www.googleapis.com/auth/drive.file"
APPLICATION_NAME = "Drive API Python Quickstart"
DEBUG = False


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser("~")
    credential_dir = os.path.join(home_dir, ".credentials")
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir, "ml_template_gdrive.json")

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        flags = DotMap()
        flags.noauth_local_webserver = True
        flags.logging_level = "ERROR"
        credentials = tools.run_flow(flow, store, flags)
        if DEBUG:
            print("Storing credentials to " + credential_path)
    return credentials


def delete_item(service, item):
    assert item["name"].endswith(".ipynb")
    print("    Deleting: {0} ({1})".format(item["name"], item["id"]))
    try:
        service.files().delete(fileId=item["id"]).execute()
    except errors.HttpError as error:
        print("    An error occurred: {}".format(error))


def delete_all_in_folder(service, folder_id):
    PAGE_SIZE = 100
    query = "'{}' in parents".format(folder_id)
    if DEBUG:
        print("Query:", query)
    results = service.files().list(
        spaces="drive", q=query, pageSize=PAGE_SIZE, fields="nextPageToken, files(id, name)").execute()
    items = results.get("files", [])
    if not items:
        print("  No files found.")
    else:
        print("  Files:")
        for item in items:
            delete_item(service, item)
        print()


def upload_notebook(service, f, folder_id):
    print("  Uploading {}".format(f))

    notebook_name = os.path.basename(f)
    file_metadata = {"name": notebook_name, "mimeType": "application/vnd.google.colab", "parents": [folder_id]}
    media = MediaFileUpload(f, mimetype="application/x-ipynb+json", resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    print("    File ID: {}".format(file.get("id")))


def main(notebook_folder, folder_id):
    if DEBUG:
        print("Notebook folder:", os.listdir(notebook_folder))
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build("drive", "v3", http=http)

    print("Deleting old notebook versions...")
    delete_all_in_folder(service, folder_id)

    print("Uploading new notebook versions...")
    for f in os.listdir(notebook_folder):
        f = os.path.join(notebook_folder, f)
        if os.path.isfile(f) and f.endswith(".ipynb"):
            upload_notebook(service, f, folder_id)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("notebookFolder")
    parser.add_argument("folderId")
    parser.add_argument("clientSecret")
    args = parser.parse_args()

    global CLIENT_SECRET_FILE
    if args.clientSecret is not None:
        CLIENT_SECRET_FILE = args.clientSecret
    else:
        CLIENT_SECRET_FILE = "client_secret.json"

    main(args.notebookFolder, args.folderId)
