"""Phase 1: look inside the exported graphs.

1. I/O signature (shows where dynamic_axes left a symbol instead of a number)
2. node histogram by op type, static vs dynamic vs dynamo
3. the exact nodes dynamic_axes added
4. the tail of the graph: how the [B,84,8400] output is assembled
"""
import collections, onnx

def load(p):
    return onnx.load(p)

def dims(vi):
    return [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim]

def ops(m):
    return collections.Counter(n.op_type for n in m.graph.node)

graphs = {k: load(f"models/{f}") for k, f in
          [("static", "yolov8n_static.onnx"), ("dynamic", "yolov8n.onnx"), ("dynamo", "yolov8n_dynamo.onnx")]}

print("== 1. I/O signature ==")
for k, m in graphs.items():
    print(f"{k:8s} opset={m.opset_import[0].version:<3d} nodes={len(m.graph.node):<4d} "
          f"initializers={len(m.graph.initializer):<4d}"
          f" in={[(i.name, dims(i)) for i in m.graph.input]} out={[(o.name, dims(o)) for o in m.graph.output]}")

print("\n== 2. op histogram ==")
all_ops = sorted(set().union(*[ops(m) for m in graphs.values()]))
print(f"{'op':22s}" + "".join(f"{k:>9s}" for k in graphs))
for op in all_ops:
    print(f"{op:22s}" + "".join(f"{ops(m)[op]:9d}" for m in graphs.values()))

print("\n== 3. nodes present ONLY in the dynamic-batch legacy graph ==")
static_names = {n.name for n in graphs["static"].graph.node}
for n in graphs["dynamic"].graph.node:
    if n.name not in static_names:
        print(f"  {n.op_type:10s} {n.name:45s} in={list(n.input)} out={list(n.output)}")

print("\n== 3b. where does the target shape of the DFL Reshape come from? ==")
for k in ("static", "dynamic"):
    g = graphs[k].graph
    producer = {o: n for n in g.node for o in n.output}
    r = next(n for n in g.node if n.op_type == "Reshape" and "dfl" in n.name)
    src = producer.get(r.input[1])
    val = onnx.numpy_helper.to_array(src.attribute[0].t).tolist() if src.op_type == "Constant" else None
    print(f"  {k:8s} {r.name}: shape input produced by {src.op_type} {src.name}"
          + (f" = {val}" if val is not None else f" <- inputs {list(src.input)}"))

print("\n== 4. last 25 nodes of the main graph (the Detect head's decode) ==")
for n in graphs["dynamic"].graph.node[-25:]:
    print(f"  {n.op_type:10s} {n.name}")
