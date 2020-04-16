import math

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


def create_data_model(xy):
    """Stores the data for the problem."""
    data = {}
    # Locations in block units
    data['locations'] = xy
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


# Check if this can't be made faster with scipy distance matrix
def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))))
    return distances


def tsp(xy):
    """Entry point of the program."""
    # Instantiate the data problem.

    xyt = list(map(tuple, xy))
    data = create_data_model(xyt)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = compute_euclidean_distance_matrix(data['locations'])

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # search_parameters.time_limit.seconds = 30
    # search_parameters.log_search = True

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    index = routing.Start(0)
    sln = []

    while not routing.IsEnd(index):
        sln.append(manager.IndexToNode(index))
        previous_index = index
        index = assignment.Value(routing.NextVar(index))

    sln.append(0)

    return sln
