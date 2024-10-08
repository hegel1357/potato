from graphviz import Digraph

def create_detailed_regnet_y_400mf_diagram():
    dot = Digraph(comment='Detailed RegNetY-400MF Architecture')
    dot.attr(rankdir='TB', size='24,24', splines='false', fontsize='20')  # Increase size for better clarity

    # Define node styles with larger font size
    dot.attr('node', shape='box', style='filled', color='black', fontcolor='black', fillcolor='white', fontsize='16')

    # Define edge styles with larger arrow size
    dot.attr('edge', arrowhead='normal', arrowsize='0.5', minlen='1')  # Adjust arrow size and edge length

    # Input
    dot.node('input', 'Input Image\n3x224x224')

    # Stem
    dot.node('stem', 'Stem\n3x3 Conv, BN, ReLU\n32 channels')

    # Stages with details
    dot.node('stage1', 'AnyStage 1\n1 block\nstride=2\n48 channels')
    dot.node('stage2', 'AnyStage 2\n3 blocks\nstride=2\n104 channels')
    dot.node('stage3', 'AnyStage 3\n6 blocks\nstride=2\n208 channels')
    dot.node('stage4', 'AnyStage 4\n16 blocks\nstride=2\n440 channels')

    # Head
    dot.node('avgpool', 'Global Average Pooling')
    dot.node('fc', 'Fully Connected\n440 -> num_classes')

    # Output
    dot.node('output', 'Output\nnum_classes')

    # Connect nodes
    dot.edge('input', 'stem')
    dot.edge('stem', 'stage1')
    dot.edge('stage1', 'stage2')
    dot.edge('stage2', 'stage3')
    dot.edge('stage3', 'stage4')
    dot.edge('stage4', 'avgpool')
    dot.edge('avgpool', 'fc')
    dot.edge('fc', 'output')

    return dot

# Generate and save the diagram as SVG
diagram = create_detailed_regnet_y_400mf_diagram()
diagram.render('detailed_regnet_y_400mf_architecture', format='svg', cleanup=True)
print("Diagram saved as 'detailed_regnet_y_400mf_architecture.svg'")
