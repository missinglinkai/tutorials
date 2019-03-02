"""
This is a script to test network access to the MissingLink deep learning tools.
"""

import pprint
from urllib.request import urlopen, Request
import urllib
import sys

RESULTS_FNAME = 'results.txt'

TESTS = [
    {
        "url": "https://missinglink.ai",
        "note": "web: required, ml: required",
    },
    {
        "url": "https://missinglink.ai/console/auth",
        "note": "web: required, ml: required, ",
    },
    {
        "url": "https://missinglinkai.appspot.com",
        "note": "[web, sdk, ml], required for communicating with MissingLink backend.",
    },
    {
        "url": "https://pypi.python.org",
        "note": "sdk: required, This is where we check if there is a new updated package to install, if this address is not accessible self update of the SDK will not work.",
    },
    {
        "url": "https://www.googleapis.com",
        "note": "[web: required] All services uses this domain as its the where the API is hosted on",
    },
    {
        # "url": "https://*.firebaseio.com",
        "url": "https://s-usc1c-nss-265.firebaseio.com",
        "note": "[web,required] This is where google identity toolkit API is hosted, we convert our authentication token to a token that can access firebase."
    },
    {
        "url": "https://api.segment.io",
        "note": "[web, optional], analytics",
    },
    {
        "url": "https://rs.fullstory.com",
        "note": "[web, optional], analytics"
    },
    {
        # "url": "https://*.intercom.io",
        "url": "https://nexus-websocket-a.intercom.io",
        "note": "[web, optional], contact support",
    },
    {
        # "url": "https://*.intercom.io",
        "url": "https://nexus-websocket-b.intercom.io",
        "note": "[web, optional], contact support",
    },
]

EXPECT = {
    'https://api.segment.io': {'code': 404, 'result': b'404 page not found\n'},
    'https://missinglink.ai': {'code': 200,
                                'result': b'<!doctype html>\n<html lang="en-US" p'
                                b'refix="og: htt'},
    'https://missinglink.ai/console/auth': {'code': 200,
                                            'result': b'<!DOCTYPE html>\n<html la'
                                            b'ng="en">\n<head>\n    <tit'
                                            b'le'},
    'https://missinglinkai.appspot.com': {'code': 200, 'result': b'ok'},
    'https://nexus-websocket-a.intercom.io': {'code': 404,
                                              'result': b'404 page not found\n'},
    'https://nexus-websocket-b.intercom.io': {'code': 404,
                                              'result': b'404 page not found\n'},
    'https://pypi.python.org': {'code': 200,
                                'result': b'\n\n\n\n\n\n\n\n<!DOCTYPE html>\n<htm'
                                b'l lang="en">\n  <head>\n'},
    'https://rs.fullstory.com': {'code': 404, 'result': b'404 page not found\n'},
    'https://s-usc1c-nss-265.firebaseio.com': {'code': 200,
                                               'result': b'\n<!DOCTYPE html>\n<ht'
                                               b'ml lang="en">\n  <hea'
                                               b'd>\n  <meta'},
    'https://www.googleapis.com': {'code': 404, 'result': b'Not Found'}
}


def get(url):
    trunc = 50
    try:
        response = urlopen(Request(url, headers={'User-Agent': 'Mozilla'}))
        code = response.code
        result = response.read()
    except urllib.error.HTTPError as err:
        code = err.code
        result = err.read()
    except Exception as err:
        return {
            "code": "invalid exception",
            "result": str(err),
        }
    #print(f"{code} {url} {len(result)} len, result: {result[:trunc]}{'...' if len(result) > trunc else ''}")
    got = {
        "code": code,
        "result": result[:trunc],
    }
    return got


def main():
    all_results = {}
    any_failed = False
    for test in TESTS:
        url = test["url"]
        note = test["note"]
        results = get(url)
        all_results[url] = results
        expected_results = EXPECT[url]
        if results == expected_results:
            print(f"Works as intended: {url}")
        else:
            any_failed = True
            print(f"We have a problem with {url}")
            print(f"  Expected: {expected_results}")
            print(f"  Got: {results}")
            print(f"Note for this URL: {note}")
    save_results(all_results)
    if any_failed:
        print("*" * 40)
        print("It seems you have at least one failed network test.")
        print(f"We've generated a file with the test results at '{RESULTS_FNAME}'.")
        print(f"Please send an email to support@missinglink.ai with the '{RESULTS_FNAME}' file attached.")
        print("*" * 40)
    else:
        print("All network tests have passed successfully.")



def save_results(results):
    with open(RESULTS_FNAME, 'w') as results_out:
        results_out.write(pprint.pformat(results))


if __name__ == "__main__":
    main()
