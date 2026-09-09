from ULSG_US_S_nomx import ULSG_US_S

for point_id in (1, 2, 3, 4):
    proj = ULSG_US_S.Projection(point_id)
    proj.result_av()
    proj.result_guar()
    proj.result_cf()
