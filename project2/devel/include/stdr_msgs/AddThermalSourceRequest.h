// Generated by gencpp from file stdr_msgs/AddThermalSourceRequest.msg
// DO NOT EDIT!


#ifndef STDR_MSGS_MESSAGE_ADDTHERMALSOURCEREQUEST_H
#define STDR_MSGS_MESSAGE_ADDTHERMALSOURCEREQUEST_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <stdr_msgs/ThermalSource.h>

namespace stdr_msgs
{
template <class ContainerAllocator>
struct AddThermalSourceRequest_
{
  typedef AddThermalSourceRequest_<ContainerAllocator> Type;

  AddThermalSourceRequest_()
    : newSource()  {
    }
  AddThermalSourceRequest_(const ContainerAllocator& _alloc)
    : newSource(_alloc)  {
  (void)_alloc;
    }



   typedef  ::stdr_msgs::ThermalSource_<ContainerAllocator>  _newSource_type;
  _newSource_type newSource;





  typedef boost::shared_ptr< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> const> ConstPtr;

}; // struct AddThermalSourceRequest_

typedef ::stdr_msgs::AddThermalSourceRequest_<std::allocator<void> > AddThermalSourceRequest;

typedef boost::shared_ptr< ::stdr_msgs::AddThermalSourceRequest > AddThermalSourceRequestPtr;
typedef boost::shared_ptr< ::stdr_msgs::AddThermalSourceRequest const> AddThermalSourceRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator1> & lhs, const ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator2> & rhs)
{
  return lhs.newSource == rhs.newSource;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator1> & lhs, const ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace stdr_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "4c174e42cb7f2369736da76a09bfbaae";
  }

  static const char* value(const ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x4c174e42cb7f2369ULL;
  static const uint64_t static_value2 = 0x736da76a09bfbaaeULL;
};

template<class ContainerAllocator>
struct DataType< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "stdr_msgs/AddThermalSourceRequest";
  }

  static const char* value(const ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "stdr_msgs/ThermalSource newSource\n"
"\n"
"================================================================================\n"
"MSG: stdr_msgs/ThermalSource\n"
"# Source description\n"
"\n"
"string id\n"
"float32 degrees\n"
"\n"
"# sensor pose, relative to the map origin\n"
"geometry_msgs/Pose2D pose \n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Pose2D\n"
"# Deprecated\n"
"# Please use the full 3D pose.\n"
"\n"
"# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.\n"
"\n"
"# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.\n"
"\n"
"\n"
"# This expresses a position and orientation on a 2D manifold.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 theta\n"
;
  }

  static const char* value(const ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.newSource);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct AddThermalSourceRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::stdr_msgs::AddThermalSourceRequest_<ContainerAllocator>& v)
  {
    s << indent << "newSource: ";
    s << std::endl;
    Printer< ::stdr_msgs::ThermalSource_<ContainerAllocator> >::stream(s, indent + "  ", v.newSource);
  }
};

} // namespace message_operations
} // namespace ros

#endif // STDR_MSGS_MESSAGE_ADDTHERMALSOURCEREQUEST_H
