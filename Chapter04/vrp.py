import random

import numpy as np
import matplotlib.pyplot as plt

import tsp


class VehicleRoutingProblem:

    def __init__(self, tspName, numOfVehicles, depotIndex):
        """
        Creates an instance of a VRP
        :param tspName: name of the underlying TSP
        :param numOfVehicles: number of vehicles used
        :param depotIndex: the index of the TSP city that will be used as the depot location
        """
        self.tsp = tsp.TravelingSalesmanProblem(tspName)
        self.numOfVehicles = numOfVehicles
        self.depotIndex = depotIndex

    def __len__(self):
        """
        returns the number of indices used to internally represent the VRP
        :return: the number of indices used to internally represent the VRP
        """
        return len(self.tsp) + self.numOfVehicles - 1

    def getRoutes(self, indices):
        """
        breaks the list of given indices into separate routes,
        by detecting the 'separator' indices
        :param indices: list of indices, including 'separator' indices
        :return: a list of routes, each route being a list of location indices from the tsp problem
        """

        # initialize lists:
        routes = []
        route = []

        # loop over all indices in the list:
        for i in indices:

            # skip depot index:
            if i == self.depotIndex:
                continue

            # index is part of the current route:
            if not self.isSeparatorIndex(i):
                route.append(i)

            # separator index - route is complete:
            else:
                routes.append(route)
                route = []  # reset route

        # append the last route:
        if route or self.isSeparatorIndex(i):
            routes.append(route)

        return routes

    def isSeparatorIndex(self, index):
        """
        Finds if curent index is a separator index
        :param index: denotes the index of the location
        :return: True if the given index is a separator
        """
        # check if the index is larger than the number of the participating locations:
        return index >= len(self) - (self.numOfVehicles - 1)

    def getRouteDistance(self, indices):
        """Calculates total the distance of the path that starts at the depo location and goes through
        the cities described by the given indices

        :param indices: a list of ordered city indices describing the given path.
        :return: total distance of the path described by the given indices
        """
        if not indices:
            return 0

        # find the distance between the depo location and the city:
        distance = self.tsp.distances[self.depotIndex][indices[0]]

        # add the distance between the last city and the depot location:
        distance += self.tsp.distances[indices[-1]][self.depotIndex]

        # add the distances between the cities along the route:
        for i in range(len(indices) - 1):
            distance += self.tsp.distances[indices[i]][indices[i + 1]]
        return distance

    def getTotalDistance(self, indices):
        """Calculates the combined distance of the various paths described by the given indices

        :param indices: a list of ordered city indices and separator indices describing one or more paths.
        :return: combined distance of the various paths described by the given indices
        """
        totalDistance = 0
        for route in self.getRoutes(indices):
            routeDistance = self.getRouteDistance(route)
            #print("- route distance = ", routeDistance)
            totalDistance += routeDistance
        return totalDistance

    def getMaxDistance(self, indices):
        """Calculates the max distance among the distances of the various paths described by the given indices

        :param indices: a list of ordered city indices and separator indices describing one or more paths.
        :return: max distance among the distances of the various paths described by the given indices
        """
        maxDistance = 0
        for route in self.getRoutes(indices):
            routeDistance = self.getRouteDistance(route)
            #print("- route distance = ", routeDistance)
            maxDistance = max(routeDistance, maxDistance)
        return maxDistance

    def getAvgDistance(self, indices):
        """Calculates the average distance among the distances of the various paths described by the given indices
        Does not consider empty paths

        :param indices: a list of ordered city indices and separator indices describing one or more paths.
        :return: max distance among the distances of the various paths described by the given indices
        """

        routes = self.getRoutes(indices)
        totalDistance = 0
        counter = 0
        for route in routes:
            if route:  # consider only routes that are not empty
                routeDistance = self.getRouteDistance(route)
                # print("- route distance = ", routeDistance)
                totalDistance += routeDistance
                counter += 1
        return totalDistance/counter

    def plotData(self, indices):
        """breaks the list of indices into separate routes and plot each route in a different color

        :param indices: A list of ordered indices describing the combined routes
        :return: the resulting plot
        """

        # plot th ecities of the underlying TSP:
        plt.scatter(*zip(*self.tsp.locations), marker='.', color='red')

        # mark the depot location with a large 'X':
        d = self.tsp.locations[self.depotIndex]
        plt.plot(d[0], d[1], marker='x', markersize=10, color='green')

        # break the indices to separate routes and plot each route in a different color:
        routes = self.getRoutes(indices)
        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.numOfVehicles)))
        for route in routes:
            route = [self.depotIndex] + route + [self.depotIndex]
            stops = [self.tsp.locations[i] for i in route]
            plt.plot(*zip(*stops), linestyle='-', color=next(color))

        return plt


def main():
    # create a problem instance:
    vrp = VehicleRoutingProblem("bayg29", 3, 12)

    # generate random solution and evaluate it:
    randomSolution = random.sample(range(len(vrp)), len(vrp))
    print("random solution = ", randomSolution)
    print("route breakdown = ", vrp.getRoutes(randomSolution))
    print("max distance = ", vrp.getMaxDistance(randomSolution))

    # plot the solution:
    plot = vrp.plotData(randomSolution)
    plot.show()


if __name__ == "__main__":
    main()

