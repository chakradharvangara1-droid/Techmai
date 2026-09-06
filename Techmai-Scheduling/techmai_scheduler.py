"""Techmai weather-aware scheduling prototype using CSV files and an AI agent."""

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
	from openai import OpenAI
except ImportError:
	OpenAI = None


BASE_DIR = Path(__file__).resolve().parent
ORDERS_FILE = BASE_DIR / "orders.csv"
TECHNICIANS_FILE = BASE_DIR / "technicians.csv"
POLICY_DIR = BASE_DIR / "policies"


TIME_WINDOWS = {
	"1": ("08:00", "12:00", "8 AM - 12 PM"),
	"2": ("12:00", "16:00", "12 PM - 4 PM"),
	"3": ("16:00", "20:00", "4 PM - 8 PM"),
}


@dataclass
class Order:
	order_id: str
	customer: str
	city: str
	service_type: str
	scheduled_date: date
	time_window: str
	technician: str


@dataclass
class Weather:
	city: str
	forecast_date: date
	condition: str
	temperature_c: int
	wind_kmh: int
	precipitation_probability: int


@dataclass
class Technician:
	name: str
	cities: tuple[str, ...]
	skills: tuple[str, ...]
	unavailable: set[tuple[date, str]]


def load_orders() -> list[Order]:
	with ORDERS_FILE.open(newline="", encoding="utf-8") as file:
		return [
			Order(
				row["order_id"], row["customer"], row["city"], row["service_type"],
				parse_date(row["scheduled_date"]), row["time_window"], row["technician"],
			)
			for row in csv.DictReader(file)
		]


def load_technicians() -> list[Technician]:
	with TECHNICIANS_FILE.open(newline="", encoding="utf-8") as file:
		technicians = []
		for row in csv.DictReader(file):
			unavailable = set()
			for value in row["unavailable"].split("|") if row["unavailable"] else []:
				unavailable_date, window = value.split(":")
				unavailable.add((parse_date(unavailable_date), window))
			technicians.append(Technician(
				row["name"], tuple(row["cities"].split("|")),
				tuple(row["skills"].split("|")), unavailable,
			))
		return technicians


def get_weather(city: str, scheduled_date: date) -> Weather:
	"""Get a daily forecast from Open-Meteo using the city's geocoded location."""
	location_response = requests.get(
		"https://geocoding-api.open-meteo.com/v1/search",
		params={"name": city, "count": 1, "language": "en", "format": "json"},
		timeout=15,
	)
	location_response.raise_for_status()
	location = location_response.json()
	results = location.get("results", [])
	if not results:
		raise ValueError(f"City not found: {city}")

	latitude, longitude = results[0]["latitude"], results[0]["longitude"]
	forecast_response = requests.get(
		"https://api.open-meteo.com/v1/forecast",
		params={
			"latitude": latitude,
			"longitude": longitude,
			"daily": "weather_code,temperature_2m_max,wind_speed_10m_max,precipitation_probability_max",
			"temperature_unit": "celsius",
			"wind_speed_unit": "kmh",
			"timezone": "auto",
			"start_date": scheduled_date.isoformat(),
			"end_date": scheduled_date.isoformat(),
		},
		timeout=15,
	)
	forecast_response.raise_for_status()
	forecast = forecast_response.json()
	daily = forecast["daily"]
	weather_code = int(daily["weather_code"][0])
	condition = weather_condition(weather_code)
	return Weather(
		city, scheduled_date, condition,
		round(daily["temperature_2m_max"][0]),
		round(daily["wind_speed_10m_max"][0]),
		daily["precipitation_probability_max"][0] or 0,
	)


def weather_condition(code: int) -> str:
	if code in {71, 73, 75, 77, 85, 86}:
		return "Heavy snow" if code in {75, 86} else "Snow"
	if code in {56, 57, 66, 67}:
		return "Freezing rain"
	if code in {95, 96, 99}:
		return "Thunderstorm"
	if code in {51, 53, 55, 61, 63, 65, 80, 81, 82}:
		return "Rain"
	return "Clear"


def assess_weather(weather: Weather, service_type: str) -> tuple[bool, str]:
	"""Apply deterministic safety rules before any AI explanation."""
	severe_conditions = {"Heavy snow", "Freezing rain", "Ice storm", "Thunderstorm"}

	if weather.condition in severe_conditions:
		return True, f"{weather.condition} is unsafe for technician travel and field work."
	if weather.wind_kmh >= 40 and "Outdoor" in service_type:
		return True, f"Wind speed of {weather.wind_kmh} km/h is unsafe for outdoor work."
	if weather.precipitation_probability >= 80 and "Outdoor" in service_type:
		return True, f"There is an {weather.precipitation_probability}% chance of precipitation."
	return False, "Weather conditions are within the operating limits."


def find_replacement(order: Order, technicians: list[Technician]) -> tuple[date, str, str] | None:
	"""Find the first available skilled technician and appointment window."""
	for day_offset in range(1, 8):
		candidate_date = order.scheduled_date + timedelta(days=day_offset)
		for window_id in TIME_WINDOWS:
			if candidate_date == order.scheduled_date and window_id == order.time_window:
				continue
			for technician in technicians:
				can_work = (
					order.city in technician.cities
					and order.service_type in technician.skills
					and (candidate_date, window_id) not in technician.unavailable
				)
				if can_work:
					return candidate_date, window_id, technician.name
	return None


def format_date(value: date) -> str:
	return f"{value:%B} {value.day}, {value.year}"


def evaluate_order(order: Order, weather: Weather, technicians: list[Technician]) -> str:
	"""Return a readable scheduling recommendation for one order."""
	needs_change, reason = assess_weather(weather, order.service_type)
	current_window = TIME_WINDOWS[order.time_window][2]

	lines = [
		f"Order: {order.order_id} | Customer: {order.customer}",
		f"Current appointment: {format_date(order.scheduled_date)}, {current_window}",
		f"Technician: {order.technician} | Service: {order.service_type}",
		(
			f"Forecast: {weather.condition}, {weather.temperature_c} C, "
			f"wind {weather.wind_kmh} km/h, precipitation {weather.precipitation_probability}%"
		),
	]

	if not needs_change:
		lines.extend(["Decision: NO SCHEDULE CHANGE", f"Reason: {reason}"])
		return "\n".join(lines)

	replacement = find_replacement(order, technicians)
	lines.extend(["Decision: SCHEDULE CHANGE REQUIRED", f"Reason: {reason}"])
	if replacement:
		replacement_date, replacement_window, replacement_technician = replacement
		lines.append(
			f"Suggested appointment: {format_date(replacement_date)}, "
			f"{TIME_WINDOWS[replacement_window][2]} with {replacement_technician}"
		)
		lines.append("Status: Awaiting human approval")
	else:
		lines.append("Suggested appointment: No suitable slot found in the next seven days")
	return "\n".join(lines)


def parse_date(value: str) -> date:
	return datetime.strptime(value, "%Y-%m-%d").date()


def order_to_dict(order: Order) -> dict:
	result = asdict(order)
	result["scheduled_date"] = order.scheduled_date.isoformat()
	return result


def retrieve_policies(query: str, limit: int = 3) -> str:
	"""Retrieve the most relevant local policy documents for the agent."""
	query_terms = {
		term for term in re.findall(r"[a-z0-9]+", query.lower())
		if len(term) > 2
	}
	documents = []
	for path in sorted(POLICY_DIR.glob("*.md")):
		content = path.read_text(encoding="utf-8")
		document_terms = set(re.findall(r"[a-z0-9]+", content.lower()))
		score = len(query_terms & document_terms)
		documents.append((score, path.name, content))

	selected = sorted(documents, key=lambda item: (-item[0], item[1]))[:limit]
	return "\n\n".join(
		f"Source: {name}\n{content}" for score, name, content in selected if score > 0
	) or "No matching Techmai policy was found."


def check_order_weather(order_id: str) -> str:
	"""Agent tool: retrieve weather and evaluate one CSV order."""
	orders = load_orders()
	technicians = load_technicians()
	order = next((item for item in orders if item.order_id == order_id), None)
	if order is None:
		return json.dumps({"error": f"Order not found: {order_id}"})
	weather = get_weather(order.city, order.scheduled_date)
	needs_change, reason = assess_weather(weather, order.service_type)
	weather_data = asdict(weather)
	weather_data["forecast_date"] = weather.forecast_date.isoformat()
	return json.dumps({
		"order": order_to_dict(order),
		"weather": weather_data,
		"needs_schedule_change": needs_change,
		"reason": reason,
		"recommendation": evaluate_order(order, weather, technicians),
	})


def find_replacement_slot(order_id: str) -> str:
	"""Agent tool: find a compatible replacement slot from the technician CSV."""
	orders = load_orders()
	order = next((item for item in orders if item.order_id == order_id), None)
	if order is None:
		return json.dumps({"error": f"Order not found: {order_id}"})
	replacement = find_replacement(order, load_technicians())
	if replacement is None:
		return json.dumps({"order_id": order_id, "replacement": None})
	replacement_date, window, technician = replacement
	return json.dumps({
		"order_id": order_id,
		"replacement": {
			"date": replacement_date.isoformat(),
			"time_window": TIME_WINDOWS[window][2],
			"technician": technician,
		},
	})


AGENT_TOOLS = [
	{
		"type": "function",
		"function": {
			"name": "check_order_weather",
			"description": "Check the weather and schedule decision for a Techmai order.",
			"parameters": {
				"type": "object",
				"properties": {"order_id": {"type": "string"}},
				"required": ["order_id"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "find_replacement_slot",
			"description": "Find an available technician and replacement time window.",
			"parameters": {
				"type": "object",
				"properties": {"order_id": {"type": "string"}},
				"required": ["order_id"],
			},
		},
	},
]

AVAILABLE_AGENT_TOOLS = {
	"check_order_weather": check_order_weather,
	"find_replacement_slot": find_replacement_slot,
}


def run_agent(question: str) -> str:
	"""Ask Groq to coordinate the scheduling tools and summarize the result."""
	load_dotenv()
	api_key = os.getenv("GROQ_API_KEY")
	if not api_key or OpenAI is None:
		return "AI agent unavailable. Set GROQ_API_KEY and install the openai package."

	client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
	policy_context = retrieve_policies(question)
	messages = [
		{"role": "system", "content": (
			"You are Techmai's scheduling agent. Use tools to inspect orders and "
			"recommend changes. Never claim an appointment was changed; approval is required. "
			"Use the retrieved policy context as company guidance, and mention its source "
			"when explaining a recommendation.\n\n"
			f"Retrieved policy context:\n{policy_context}"
		)},
		{"role": "user", "content": question},
	]
	for _ in range(3):
		response = client.chat.completions.create(
			model="qwen/qwen3.8-27b", messages=messages, tools=AGENT_TOOLS, max_tokens=512
		)
		message = response.choices[0].message
		if not message.tool_calls:
			return message.content or "The agent did not return a response."

		messages.append(message.model_dump(exclude_none=True))
		for tool_call in message.tool_calls:
			tool = AVAILABLE_AGENT_TOOLS.get(tool_call.function.name)
			arguments = json.loads(tool_call.function.arguments)
			result = tool(**arguments) if tool else json.dumps({"error": "Unknown tool"})
			messages.append({
				"role": "tool",
				"tool_call_id": tool_call.id,
				"content": result,
			})
	return "The agent reached the tool-call limit without returning a summary."


def choose_order() -> Order:
	orders = load_orders()
	print("\nOrders loaded from orders.csv:")
	for order in orders:
		print(f"  {order.order_id}: {order.customer}, {order.city}, {order.scheduled_date}")

	order_id = input("Order ID (or NEW): ").strip().upper()
	existing = next((order for order in orders if order.order_id == order_id), None)
	if existing:
		return existing

	if order_id != "NEW":
		raise ValueError("Unknown order ID")

	city = input("Customer city: ").strip().title()
	appointment_date = parse_date(input("Appointment date (YYYY-MM-DD): ").strip())
	print("Time windows: 1) 8 AM - 12 PM  2) 12 PM - 4 PM  3) 4 PM - 8 PM")
	time_window = input("Time window: ").strip()
	if time_window not in TIME_WINDOWS:
		raise ValueError("Time window must be 1, 2, or 3")
	return Order(
		"NEW", "Test customer", city, "Outdoor installation",
		appointment_date, time_window, "Unassigned"
	)


def main() -> None:
	print("Techmai Weather-Aware Scheduling Prototype")
	print("Orders and technicians are loaded from CSV files.")
	while True:
		try:
			order = choose_order()
			weather = get_weather(order.city, order.scheduled_date)
			print("\n" + evaluate_order(order, weather, load_technicians()))
			if input("Ask the AI agent to summarize this order? (y/n): ").strip().lower() == "y":
				print("\nAgent: " + run_agent(f"Check order {order.order_id} and explain whether it needs rescheduling."))
		except ValueError as error:
			print(f"Input error: {error}")
		if input("\nCheck another order? (y/n): ").strip().lower() != "y":
			break


if __name__ == "__main__":
	main()
