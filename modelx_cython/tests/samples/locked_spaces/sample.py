from LockedSpaces_nomx import LockedSpaces

# single-threaded: the tracer records the calls of this thread only
for i in range(3):
    for t in range(5):
        LockedSpaces.Projection[i].pv(t)
