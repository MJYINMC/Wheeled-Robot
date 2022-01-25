// Generated by gencpp from file course_agv_nav/PlanRequest.msg
// DO NOT EDIT!


#ifndef COURSE_AGV_NAV_MESSAGE_PLANREQUEST_H
#define COURSE_AGV_NAV_MESSAGE_PLANREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace course_agv_nav
{
template <class ContainerAllocator>
struct PlanRequest_
{
  typedef PlanRequest_<ContainerAllocator> Type;

  PlanRequest_()
    {
    }
  PlanRequest_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }







  typedef boost::shared_ptr< ::course_agv_nav::PlanRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::course_agv_nav::PlanRequest_<ContainerAllocator> const> ConstPtr;

}; // struct PlanRequest_

typedef ::course_agv_nav::PlanRequest_<std::allocator<void> > PlanRequest;

typedef boost::shared_ptr< ::course_agv_nav::PlanRequest > PlanRequestPtr;
typedef boost::shared_ptr< ::course_agv_nav::PlanRequest const> PlanRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::course_agv_nav::PlanRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::course_agv_nav::PlanRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


} // namespace course_agv_nav

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::course_agv_nav::PlanRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::course_agv_nav::PlanRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::course_agv_nav::PlanRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::course_agv_nav::PlanRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::course_agv_nav::PlanRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::course_agv_nav::PlanRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::course_agv_nav::PlanRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d41d8cd98f00b204e9800998ecf8427e";
  }

  static const char* value(const ::course_agv_nav::PlanRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd41d8cd98f00b204ULL;
  static const uint64_t static_value2 = 0xe9800998ecf8427eULL;
};

template<class ContainerAllocator>
struct DataType< ::course_agv_nav::PlanRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "course_agv_nav/PlanRequest";
  }

  static const char* value(const ::course_agv_nav::PlanRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::course_agv_nav::PlanRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n"
;
  }

  static const char* value(const ::course_agv_nav::PlanRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::course_agv_nav::PlanRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct PlanRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::course_agv_nav::PlanRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::course_agv_nav::PlanRequest_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // COURSE_AGV_NAV_MESSAGE_PLANREQUEST_H
