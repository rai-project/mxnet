package mxnet

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/awalterschulze/gographviz"
	"gopkg.in/fatih/set.v0"
)

// color map
var fillcolors = []string{
	"#8dd3c7",
	"#fb8072",
	"#ffffb3",
	"#bebada",
	"#80b1d3",
	"#fdb462",
	"#b3de69",
	"#fccde5",
}
var edgecolors = []string{
	"#245b51",
	"#941305",
	"#999900",
	"#3b3564",
	"#275372",
	"#975102",
	"#597d1c",
	"#90094e",
}

func (g *Graph) ToDotGraph() (*gographviz.Escape, error) {
	tuples := func(s string) []string {
		re := regexp.MustCompile(`\d+`)
		return re.FindAllString(s, -1)
	}
	makeDefaultAttributes := func() map[string]string {
		return map[string]string{
			"shape":     "box",
			"fixedsize": "true",
			"width":     "1.3",
			"height":    "0.8034",
			"style":     "filled",
		}
	}

	isLikeWeight := func(name string) bool {
		if strings.HasSuffix(name, "_weight") {
			return true
		}
		if strings.HasSuffix(name, "_bias") {
			return true
		}
		if strings.HasSuffix(name, "_beta") ||
			strings.HasSuffix(name, "_gamma") ||
			strings.HasSuffix(name, "_moving_var") ||
			strings.HasSuffix(name, "_moving_mean") {
			return true
		}
		return false
	}

	hideWeights := true    // TODO: should be an option
	drawShape := true      // TODO: should be an option
	graphName := "mxnet"   // TODO: should be an option
	layout := "horizontal" // TODO: should be an option

	dg := gographviz.NewEscape()
	dg.SetName(graphName)
	dg.SetDir(true)

	dg.AddAttr(graphName, "nodesep", "1")
	dg.AddAttr(graphName, "ranksep", "1.5 equally")

	switch layout {
	case "vertical":
		dg.AddAttr(graphName, "rankdir", "TB")
	case "horizontal":
		dg.AddAttr(graphName, "rankdir", "RL")
	default:
	}

	hiddenNodes := set.NewNonTS()

	// make nodes
	for _, node := range g.Nodes {
		op := node.Op
		name := node.Name
		attrs := makeDefaultAttributes()
		label := op

		switch op {
		case "null":
			if isLikeWeight(name) {
				if hideWeights {
					hiddenNodes.Add(name)
					continue
				}
			}
			attrs["shape"] = "oval"
			attrs["fillcolor"] = fillcolors[0]
			label = name
		case "Convolution":
			if val, ok := node.Param["stride"]; ok {
				strideInfo := tuples(val)
				label = fmt.Sprintf("Convolution\n%s/%s, %s",
					strings.Join(tuples(node.Param["kernel"]), "x"),
					strings.Join(strideInfo, "x"),
					node.Param["num_filter"],
				)
			} else {
				label = fmt.Sprintf("Convolution\n%s/%s, %s",
					strings.Join(tuples(node.Param["kernel"]), "x"),
					"1",
					node.Param["num_filter"],
				)
			}
			attrs["fillcolor"] = fillcolors[1]
		case "FullyConnected":
			label = fmt.Sprintf("FullyConnected\n%s", node.Param["num_hidden"])
			attrs["fillcolor"] = fillcolors[1]
		case "BatchNorm":
			attrs["fillcolor"] = fillcolors[3]
		case "Activation", "LeakyReLU":
			label = fmt.Sprintf("%s\n%s", op, node.Param["act_type"])
			attrs["fillcolor"] = fillcolors[2]
		case "Pooling":
			if val, ok := node.Param["stride"]; ok {
				strideInfo := tuples(val)
				label = fmt.Sprintf("Pooling\n%s, %s/%s",
					node.Param["pool_type"],
					strings.Join(tuples(node.Param["kernel"]), "x"),
					strings.Join(strideInfo, "x"),
				)
			} else {
				label = fmt.Sprintf("Pooling\n%s, %s/%s",
					node.Param["pool_type"],
					strings.Join(tuples(node.Param["kernel"]), "x"),
					"1",
				)
			}
			attrs["fillcolor"] = fillcolors[4]
		case "Concat", "Flatten", "Reshape":
			attrs["fillcolor"] = fillcolors[5]
		case "Softmax":
			attrs["fillcolor"] = fillcolors[6]
		default:
			attrs["fillcolor"] = fillcolors[7]
			if op == "Custom" {
				label = node.Param["op_type"]
			}
			attrs["label"] = label
		}
		dg.AddNode(graphName, name, attrs)
	}

	// make edges
	for _, node := range g.Nodes {
		op := node.Op
		name := node.Name
		if op == "null" {
			continue
		}
		inputs := node.Inputs
		for _, item := range inputs {
			inputNode := g.Nodes[item.NodeId]
			inputName := inputNode.Name
			if hiddenNodes.Has(inputName) {
				continue
			}
			attrs := map[string]string{
				"dir":       "back",
				"arrowtail": "open",
			}
			if drawShape {
				key := inputName
				if inputNode.Op != "null" {
					key += "_output"
					if val, ok := inputNode.Param["num_outputs"]; ok {
						numOutputs, err := strconv.Atoi(val)
						if err != nil {
							continue
						}
						key += strconv.Itoa(numOutputs - 1)
						inputNode.Param["num_outputs"] = strconv.Itoa(numOutputs - 1)
					}

				}
				//shape = shape_dict[key][1:]
				//label = "x".join([str(x) for x in shape])
				//attrs["label"] = label
			}
			dg.AddEdge(name, inputName, true, attrs)
		}

	}

	return dg, nil
}

type ModelGraphAttributes map[string][]string
