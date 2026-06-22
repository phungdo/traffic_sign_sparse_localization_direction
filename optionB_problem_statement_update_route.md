# AutoTS Operational Critique: Q&A Log

## Q1. Does AutoTS output anything that can update a road network?

**Question:** AutoTS only returns coordinates and relative sign orientation. It does not write to any road network `way_id`. Am I right?

**Answer:** Yes.

AutoTS maps image sequences and GPS inputs `(Is, Gs)` to a tuple `(ls, os, cs)`: clustered lat/lon, orientation relative to the ego vehicle (left, right, backward), and sign category. The authors equate "backward-facing" with "belongs to the road driven on," but they never run a map-matching step to bind that coordinate to a specific `way_id` on a topological road graph.

This matters in operations because a single $(x, y)$ point can sit above an overpass, beside a parallel service road, or at the mouth of a complex intersection. Without HMM-based or topological matching against the vehicle trajectory, no system can decide which road segment inherits the sign's constraint. The paper also ignores that a `way_id` is a directed vector; knowing a sign is "backward" relative to the car only helps if you also know whether the car travels in the forward or reverse direction of that vector.

The paper handles raw information extraction from pixels to 2D/3D coordinates well. The missing piece is topological map matching.

---

## Q2. Can AutoTS distinguish a No Entry sign that restricts a side branch from one that restricts straight-through traffic?

**Question:** A No Entry sign at Branch A of a four-way intersection restricts turns into A, not straight-through traffic C to B. AutoTS cannot tell the difference.

**Answer:** Correct. The paper confuses two different map data structures: link attributes (properties of a road segment) and maneuver restrictions (constraints on specific turns at a junction node).

Consider two mapping-vehicle trajectories through the same intersection:

Case 1, vehicle goes C to B (legal). The No Entry sign at Branch A sits to the left of the camera. AutoTS classifies it as "leftward" and ignores it. Consequence: the map never learns that turns from C into A are prohibited.

Case 2, vehicle goes D to B (legal right turn). The No Entry sign at Branch A faces the camera head-on. AutoTS classifies it as "backward" and concludes the vehicle's own road is restricted. Consequence: the map incorrectly marks D-to-B as No Entry.

For AutoTS to assign the sign to Branch A correctly, the mapping vehicle would have to drive into A, violating the very restriction the sign encodes. No legal driver will produce that trajectory.

Professional map-update systems handle this differently. They compute the sign's absolute geographic heading, cast a ray backward through the sign face to identify which `way_id` it intersects, then generate turn restrictions:

- `from_way: C` to `to_way: A`, restriction: `no_left_turn`
- `from_way: B` to `to_way: A`, restriction: `no_right_turn`
- `from_way: D` to `to_way: A`, restriction: `no_straight_on`

Alternatively, if enough restrictions converge on Branch A, the system promotes the attribute to `oneway=yes` on the segment itself.

AutoTS works on straight highways with no junctions. At urban intersections its orientation logic produces corrupted data.

---

## Q3. What about multi-sign clusters at the same intersection?

**Question:** Two No Entry signs sit near each other. One is slightly tilted (elliptical cross-section from the camera's perspective), the other faces the vehicle head-on (full circle). How should a decision tree handle this?

**Answer:** The system needs a five-step pipeline that combines CV output with GIS topology.

Step 1: Convert relative camera angles to absolute headings. The full-circle sign has face heading = vehicle heading + 180 degrees. The tilted sign has face heading = vehicle heading + offset alpha, estimated from the ellipse aspect ratio.

Step 2: Cluster spatially. If the two signs are within a threshold distance (say 5 meters), group them as a single sign cluster bound to one intersection node.

Step 3: Ray-cast from each sign's back face. Both rays should intersect Branch A, identifying the target `to_way`.

Step 4: For each approach way (B, C, D), compute the collision angle between the approach direction and each sign face. If the angle indicates full frontal visibility, mark the corresponding from-to pair as restricted. If the vehicle is behind the sign, skip.

```
Input: Cluster [Sign1 heading=East, Sign2 heading=South], Target: Branch A.
For each approach into the intersection (from B, C, D):

  Compute collision angle between approach vector and sign face.

  If full frontal (Sign1 faces vehicle from D):
    Record: no_straight_on from D to A.

  If angled but visible (Sign2 faces vehicle from C):
    Record: no_left_turn from C to A.

  If vehicle is behind the sign (vehicle from A exiting):
    Skip.
```

Step 5: If two or more turn restrictions target the same branch, promote to a link attribute: set `oneway = -1` or `access = no` on Branch A's `way_id`.

The paper stops at labeling signs as tilted or frontal relative to the camera. Map operations require absolute heading, topology intersection, and rule aggregation.

---

## Q4. What about sign placement laws that vary by country?

**Question:** In right-hand-drive countries, I only obey signs on my right side. If a No Entry sign at Branch A happens to face me head-on while I drive C to B, can I still go straight?

**Answer:** Yes. You can still go straight from C to B.

In countries with right-hand traffic (Vietnam, the US, most of the EU), a sign that restricts forward travel must be placed on the right side of the road or overhead. The No Entry sign in this scenario sits at the left-side mouth of Branch A, 10 meters from the centerline of the C-to-B trajectory. Any driver or rule engine can infer that it governs the left turn into A, not the straight-through path.

AutoTS applies a different rule: "sign faces camera implies sign applies to current road." Under this logic, the system would flag C-to-B as restricted, corrupting the map.

The decision tree needs a lateral-offset check before the restriction assignment:

```
Input: No Entry sign at (x,y), face pointing toward C. Vehicle trajectory C to B.
Parameter: Drive_Side = Right.

  Project sign (x,y) onto trajectory C-to-B.
  Compute lateral offset: sign is 10m LEFT of trajectory center.

  If sign is on RIGHT side + face pointing toward ego:
    Restrict straight-through.

  If sign is on LEFT side + face pointing toward ego + Drive_Side = Right:
    Sign does NOT apply to straight-through.
    Ray-cast from sign back to find Branch A.
    Record: no_left_turn from C to A.
```

---

## Q5. What happens when a new sign appears overnight and the first autonomous vehicle encounters it?

**Question:** Two new No Entry signs appear at an intersection. The HD map does not know about them. Route planning still sends vehicles through. Vehicle 1 sees the signs, deviates from its route, and sends images back to the server. Should the server update the `way_id` immediately so all subsequent routing avoids C to B?

**Answer:** The server should not commit a hard update based on a single vehicle report.

The safety principle in autonomous driving architecture is that perception overrides the prior map. If Vehicle 1's CV detects No Entry signs facing its planned path, its motion planner must refuse to proceed regardless of what the route says. If the CV fails (as AutoTS's logic would in many junction geometries), the vehicle enters a restricted road.

When Vehicle 1 deviates, it packages an anomaly event: the planned route, the actual GPS trajectory showing the divergence, and perception logs including sign images, coordinates, and the affected `way_id`. It sends this to the cloud.

The cloud should not cut the graph edge based on one report. Vehicle 1 might have misidentified a billboard, hit a glare artifact, or encountered temporary construction signage removed an hour later. Blocking a major road on a single false positive can cascade into a routing blackhole across the city.

Instead, the cloud runs a three-phase consensus protocol:

Phase 1, broadcast warning. The server drops a low-confidence flag on the intersection node. Vehicles routed through C-to-B still receive that route, but with metadata indicating reduced reliability. Autonomous vehicles raise sensor sensitivity when approaching.

Phase 2, consensus gathering. Vehicle 2 arrives, also detects the signs, also deviates. Vehicle 3 does the same. Each sends an anomaly event.

Phase 3, hard update. When the confidence score crosses a threshold (for example, three vehicles within 30 minutes all confirming the restriction with no vehicle passing through), the graph database commits: `way_id_CB` gets `access=no`. From that point, the route engine drops the C-to-B edge entirely. All subsequent route requests steer around the intersection by default.

Because AutoTS only outputs bare coordinates without `way_id` binding, the cloud would receive thousands of $(x,y)$ points from hundreds of vehicles clustered at the intersection, with no way to determine whether those signs restrict C-to-B or D-to-A. The consensus algorithm cannot run without topological context.

---

## Q6. What about rain and snow? How fast must the Map API update?

**Question:** All the above assumed clear weather. In rain or snow, even distinguishing the sign is hard, let alone its orientation. That orientation determines the next 10 seconds of driving. How fast does the Map API need to push a new global route to all clients?

**Answer:** At 50 km/h, a vehicle covers 14 meters per second. In 10 seconds it travels 140 meters, roughly the stopping distance and the planning horizon for lane changes at urban speeds.

Rain reduces contrast, snow occludes sign faces. The camera cannot distinguish a circle from an ellipse, and may fail to read the sign content at all. When detection confidence drops below an operational threshold, the edge AI on the vehicle disables sign recognition and falls back entirely on the preloaded HD Map. If that map is stale (the No Entry signs were installed yesterday), the vehicle proceeds through the restricted intersection with full confidence in an outdated prior.

Camera failure in bad weather transfers all safety responsibility to the speed of cloud map updates. Industrial map engines split this into three tiers:

Tier 1, seconds to minutes. Pub/Sub streaming (Kafka-style). As soon as Vehicle 1 reports the anomaly, the server pushes a metadata alert to all devices within a radius (similar to Waze hazard alerts). The global route does not change, but vehicles receive a slow-down instruction and heightened sensor sensitivity for that intersection.

Tier 2, 15 minutes to one hour. Partial graph recompilation. After the consensus algorithm confirms the restriction, the graph database severs the C-to-B edge. All new route API responses exclude that path. Map tiles on screen may still show the old geometry until the next render cycle.

Tier 3, daily to weekly. Full HD map build. Thousands of vehicle traces are processed into precise 3D sign coordinates, lane topology updates, and turn restriction records. The resulting multi-gigabyte map package ships to vehicles via over-the-air update overnight.

The paper treats sign orientation classification as the end goal. In production, that classification is the most fragile input, the one most easily defeated by weather. The actual safety mechanism is the cloud pipeline that aggregates fleet observations, edits the routing graph, and broadcasts new routes to millions of devices within minutes.
