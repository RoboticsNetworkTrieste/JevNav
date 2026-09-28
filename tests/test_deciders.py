import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from jevnav.costmap import OBSTACLE, TOO_CLOSE, CostmapSpec, build_costmap
from jevnav.deciders import HeuristicDecider, JevDecider, JevUnavailable
from jevnav.kinematics import rollout
from jevnav.prompt import compose
from jevnav.route import Route

from .conftest import wall_scan


@pytest.fixture
def close_wall_request(commands):
    route = Route.through([[0.0, 0.0], [0.0, 3.0]])
    costmap = build_costmap(CostmapSpec(), [0.0, 0.0, 0.0], wall_scan(0.5), route, None)
    return compose(3, 0, costmap, commands, 1.7, np.random.default_rng(0))


def test_heuristic_never_drives_into_the_wall_and_follows_the_route(close_wall_request):
    decision = HeuristicDecider().decide(close_wall_request)
    command = close_wall_request.command(decision.command_id)
    path = rollout([0, 0, 0], command.linear, command.angular, 1.7, 0.1)
    layers = close_wall_request.costmap.layer_at(path[1:, :2])
    assert not np.isin(layers, (OBSTACLE, TOO_CLOSE)).any()
    assert decision.probabilities["F"] < 1e-6
    assert path[-1, 1] > 0.1
    assert sum(decision.probabilities.values()) == pytest.approx(1.0)


class StubServer:
    def __init__(self, answer):
        self.answer = answer
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                stub.requests.append(json.loads(self.rfile.read(length)))
                self._reply(stub.answer(stub.requests[-1]))

            def do_GET(self):
                self._reply({"data": [{"id": "rizzo-stub"}]})

            def _reply(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()


def systemone_answer(choice):
    def answer(body):
        criteria = body["questions"]["command"]["criteria"]
        probabilities = {
            key: (0.9 if key == choice else 0.1 / (len(criteria) - 1)) for key in criteria
        }
        return {
            "model": "rizzo-stub",
            "answers": {
                "command": {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": probabilities,
                    "confidence": 0.9,
                }
            },
            "usage": {"input_tokens": 955, "output_tokens": 0},
        }

    return answer


def test_jev_decider_speaks_the_systemone_wire_format(close_wall_request):
    stub = StubServer(systemone_answer("FL30"))
    try:
        decider = JevDecider(stub.url)
        assert decider.served_models() == ["rizzo-stub"]
        decision = decider.decide(close_wall_request)
    finally:
        stub.close()
    body = stub.requests[0]
    question = body["questions"]["command"]
    assert body["model"] == "rizzo-latest"
    assert body["state"] == close_wall_request.evidence
    assert question["type"] == "choice"
    assert list(question["criteria"]) == [command.id for command in close_wall_request.options]
    assert question["criteria"]["FL30"] == "Forward, curving left 15 degrees per second."
    assert "hold for the next 1.7 s" in question["instructions"]
    assert decision.command_id == "FL30"
    assert decision.probabilities["FL30"] == pytest.approx(0.9)
    assert decision.input_tokens == 955
    assert decision.served_by == "rizzo-stub"
    assert decision.latency > 0


def test_jev_decider_rejects_an_answer_outside_the_commands(close_wall_request):
    stub = StubServer(systemone_answer("JUMP"))
    try:
        with pytest.raises(JevUnavailable):
            JevDecider(stub.url).decide(close_wall_request)
    finally:
        stub.close()


def test_jev_decider_reports_a_missing_server(close_wall_request):
    with pytest.raises(JevUnavailable):
        JevDecider("http://127.0.0.1:9", timeout=2).decide(close_wall_request)
