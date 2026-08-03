import requests
import config


def get_weather(location: str) -> str:
    """Get the current weather for a given city or location."""
    if not config.OPENWEATHER_API_KEY:
        return "Weather lookup is unavailable: OPENWEATHER_API_KEY is not set."

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": config.OPENWEATHER_API_KEY, "units": "metric"}

    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        return f"Could not fetch weather for '{location}': {response.json().get('message', 'unknown error')}"

    data = response.json()
    desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    return f"{location}: {desc}, {temp}°C (feels like {feels_like}°C)"
