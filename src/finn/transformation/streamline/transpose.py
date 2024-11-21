# Protobuf onnx graph node type
from onnx import NodeProto, TensorProto
# Helper for creating ONNX nodes
from onnx import helper as oh  # noqa
# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# QONNX graph transformations for inferring datatypes and shapes
from qonnx.transformation.infer_shapes import InferShapes
# Gets items from protobuf by name
from qonnx.util.basic import get_by_name
# QONNX graph transformation base class
from qonnx.transformation.base import Transformation
# for warnings
import warnings

# Collapses repeated transpose operations into a single transpose operation
# having the same effect
# Moves a transpose operator past an elementwise addition
class CollapseRepeatedTranspose(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Transpose operation types
            if node.op_type == "Transpose":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)[0]
                # If Transpose is the final operation in the graph, there might
                # be no successor
                if successor is None or successor.op_type != "Transpose":
                    # Softly skip this node
                    continue

                # Get the (optional) permutation indices of the first transpose
                # in case it is a multi-axis transpose
                perm1 = get_by_name(node.attribute, "perm")
                # Convert permutation indices to list of integers
                perm1 = perm1.ints if perm1 is not None else None

                # Get the (optional) permutation indices of the second transpose
                # in case it is a multi-axis transpose
                perm2 = get_by_name(successor.attribute, "perm")
                # Convert permutation indices to list of integers
                perm2 = perm2.ints if perm2 is not None else None

                # Get the shape of the input tensor
                shape = model.get_tensor_shape(
                    node.input[0], fix_missing_init_shape=True
                )
                # List of dimension indices in order
                dims = (range(len(shape)))

                # Substitute the permutation indices by the reversed index list
                # of they are not given: This is default behavior, see the docs:
                #   https://onnx.ai/onnx/operators/onnx__Transpose.html
                perm1 = list(reversed(dims)) if perm1 is None else perm1
                perm2 = list(reversed(dims)) if perm2 is None else perm2

                # Combined permutation permutes the first permutation of the
                # dimensions according to the second permutation
                perm = [perm1[i] for i in perm2]

                # Create a new Transpose operator replacing the other two
                transpose = oh.make_node(
                    # Name of the operator type
                    "Transpose",
                    # Connect to the inputs to the first transpose
                    inputs=node.input,
                    # Connect to the outputs of the second transpose
                    outputs=successor.output,
                    # Insert the new permutation indices
                    perm=perm
                )
                # Insert the collapsed transpose operator
                graph.node.insert(index + 2, transpose)
                # Remove the two original transpose operators
                graph.node.remove(node)
                graph.node.remove(successor)
                # Track whether the graph has been modified, never resets to
                # False
                graph_modified = True
        # Need to redo the shape inference after potentially removing nodes
        model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified
