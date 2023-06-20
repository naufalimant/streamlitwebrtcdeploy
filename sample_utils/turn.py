import logging
import os

import streamlit as st
from twilio.rest import Client

logger = logging.getLogger(__name__)


@st.cache_data
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    
    try:
        os.environ['account_sid'] = 'ACe9b0279a642df6951ab31930eb18042b'
        os.environ['auth_token'] = '3314e95181782613a0d8e4bd1765213b'
        # account_sid = os.environ["ACe9b0279a642df6951ab31930eb18042b"]
        # auth_token = os.environ["9e395b6c8c75c454273cc75aff7e92d0"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(os.environ['account_sid'], os.environ['auth_token'])

    token = client.tokens.create()

    return token.ice_servers
