from bp.database import db_session, GtuAffiliation

with db_session() as sess:
    val = sess.query(GtuAffiliation.gtu_id).filter(GtuAffiliation.affiliation_id == 1).limit(6000).all()
    train = sess.query(GtuAffiliation.gtu_id).filter(GtuAffiliation.affiliation_id == 2).limit(18000).all()
    test = sess.query(GtuAffiliation.gtu_id).filter(GtuAffiliation.affiliation_id == 3).limit(6000).all()

val = set([i[0] for i in val])
train = set([i[0] for i in train])
test = set([i[0] for i in test])


if bool(val & train):
    val.difference_update(train)

if bool(val & test):
    test.difference_update(val)

if bool(test & train):
    test.difference_update(train)

train = list(train)
val = list(val)
test = list(test)

with db_session() as sess:
    for i in val:
        gtu_aff = GtuAffiliation(affiliation_id=5445, gtu_id=i, is_active=True)
        sess.add(gtu_aff)

    for i in train:
        gtu_aff = GtuAffiliation(affiliation_id=5445, gtu_id=i, is_active=True)
        sess.add(gtu_aff)

    for i in test:
        gtu_aff = GtuAffiliation(affiliation_id=5445, gtu_id=i, is_active=True)
        sess.add(gtu_aff)