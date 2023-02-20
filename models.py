from sqlalchemy import Boolean, Column, Integer, String

from database import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    private = Column(String, unique=True)
    owner = Column(String, unique=True, index=True)
    name = Column(String)
    is_active = Column(Boolean, default=True)
