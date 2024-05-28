from http.cookies import SimpleCookie


class SunoCookie(object):
    """
    Class wrap the interaction with Suno session_id and cookie.
    This class is originated from
    https://github.com/SunoAI-API/Suno-API/blob/main/cookie.py
    """

    def __init__(self):
        self.cookie = SimpleCookie()
        self.session_id = None
        self.token = None

    def load_cookie(self, cookie_str):
        self.cookie.load(cookie_str)

    def get_cookie(self):
        return ";".join([f"{i}={self.cookie.get(i).value}" for i in self.cookie.keys()])

    def set_session_id(self, session_id: str):
        self.session_id = session_id

    def get_session_id(self):
        return self.session_id

    def get_token(self):
        return self.token

    def set_token(self, token: str):
        self.token = token
