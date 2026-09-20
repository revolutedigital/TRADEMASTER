"""The official cTrader Open API messages (spotware/openapi-proto-messages, MIT).

The four `.proto` files next to this module are copies of the upstream files, and the `*_pb2.py`
modules are generated from them; do not edit either by hand. To regenerate (needs `protoc` and a
`protobuf` runtime at least as new as the compiler, see the `research` extra in pyproject.toml):

    cd backend/app/fx/runner/ctrader/proto
    protoc -I. --python_out=. OpenApiCommonMessages.proto OpenApiCommonModelMessages.proto \\
        OpenApiMessages.proto OpenApiModelMessages.proto
    sed -i 's/^import \\(OpenApi[A-Za-z]*_pb2\\) as /from . import \\1 as /' *_pb2.py
    sed -i '1a # fmt: off' *_pb2.py

The `sed` steps make the generated modules import each other relatively (protoc emits top-level
imports) and keep the formatter away from generated code.

SHA-256 of the files this package was generated from:

9816cd24b340dcc4eb28548eb4dd16735995d2a61889337591e5c4d8021652a2  OpenApiCommonMessages.proto
b95d7df670a7e890a53ec08f676198ace7bb0a074a4b07ff0b493c4be00a0dea  OpenApiCommonModelMessages.proto
a84df9b528e69a494e197d48e21d6291b3d76db31396663a8188721eef9fcf35  OpenApiMessages.proto
56338dcac45a149227678b7c23637d64f4e2b607f6649af82394ce5c0957fedf  OpenApiModelMessages.proto
"""
